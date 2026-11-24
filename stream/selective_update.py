"""Which inside-AABB gaussians the NTC updates in a frame (torch only; no 3DGStream imports).

Modes
  A  all gaussians inside the AABB (unchanged 3DGStream)
  B  motion-history freezing (CPOStream-style): frames 1..warmup and every `recheck`-th frame update
     everything; otherwise only gaussians whose recent observed motion exceeds a threshold
       score_i = max over the last `window` frames in which i was updated of |d_xyz_i|  (and angle)
       active  = score_xyz > tau_xyz  or  score_rot > tau_rot_deg
  C  frame-difference mask + center projection (DecoGS-style):
       per train view  M_v = dilate( max_rgb |I_t - I_{t-1}| > tau/255 , radius r )
       active_i = #views v with center_i projected inside the image, in front of the camera and on M_v >= kappa
  E  adaptive AABB (image-difference driven), recomputed every N streamed frames from the PAST window only:
       U_v   = OR over the window's frame pairs of M_v (same mask as C)
       S     = gaussians whose center hits U_v in >= kappa views
       box   = per-axis [q_lo, q_hi] percentile of S's centers, widened by `margin` x extent on each side
       active_i = center_i inside box   (the NTC bounds/hash grid stay the fixed AABB: the box is a mask only)
  EB E then B: inside the adaptive box, B's motion-history freezing (B's full frames update the whole box)
"""
import torch
import torch.nn.functional as F

MODES = ("A", "B", "C", "E", "EB")


# ----------------------------------------------------------------------------------------- mode C
def change_mask(cur, prev, tau=25.0, radius=48):
    """cur/prev: (3,H,W) float in [0,1] or uint8. Returns (H,W) bool."""
    if cur.dtype == torch.uint8:
        cur = cur.float() / 255.0
    if prev.dtype == torch.uint8:
        prev = prev.float() / 255.0
    diff = (cur - prev).abs().amax(0)
    m = (diff > tau / 255.0).float()
    if radius > 0 and m.any():
        k = 2 * radius + 1                       # square (max-pool, as in DecoGS) dilation, done separably (O(HW*k), not O(HW*k^2))
        m = F.max_pool2d(m[None, None], kernel_size=(1, k), stride=1, padding=(0, radius))
        m = F.max_pool2d(m, kernel_size=(k, 1), stride=1, padding=(radius, 0))[0, 0]
    return m > 0.5


def project_hits(xyz, full_proj, world_view, mask, znear=0.2):
    """Row-vector convention of graphdeco/3DGStream cameras (p_clip = [x,1] @ full_proj_transform).
    Returns bool (N,) : center lands on a True pixel of `mask` and is in front of the camera."""
    H, W = mask.shape
    xyz = xyz.to(full_proj.dtype)
    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    ph = torch.cat([xyz, ones], 1)
    clip = ph @ full_proj
    view_z = (ph @ world_view)[:, 2]
    w = clip[:, 3:4]
    ok = (view_z > znear) & (w[:, 0].abs() > 1e-8)
    ndc = clip[:, :2] / torch.where(w.abs() > 1e-8, w, torch.ones_like(w))
    px = ((ndc[:, 0] + 1.0) * W - 1.0) * 0.5          # graphdeco ndc2Pix
    py = ((ndc[:, 1] + 1.0) * H - 1.0) * 0.5
    xi, yi = torch.round(px).long(), torch.round(py).long()
    ok &= (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    hit = torch.zeros_like(ok)
    idx = ok.nonzero(as_tuple=True)[0]
    hit[idx] = mask[yi[idx], xi[idx]]
    return hit


def select_by_change(xyz, views, kappa=2, tau=25.0, radius=48):
    """views: iterable of (cur_img, prev_img, full_proj, world_view). Returns (active bool, stats)."""
    count = torch.zeros(xyz.shape[0], device=xyz.device, dtype=torch.int32)
    frac = []
    for cur, prev, fp, wv in views:
        m = change_mask(cur, prev, tau, radius)
        frac.append(float(m.float().mean()))
        count += project_hits(xyz, fp, wv, m).int()
    return count >= kappa, dict(mask_frac_mean=sum(frac) / max(len(frac), 1),
                                mask_frac_max=max(frac) if frac else 0.0)


# ----------------------------------------------------------------------------------------- mode B
def quat_angle_deg(q):
    """Rotation angle of (unnormalized) wxyz quaternions, degrees."""
    q = q.float()
    w = (q[:, 0].abs() / q.norm(dim=1).clamp_min(1e-12)).clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(w))


class MotionHistory:
    """Ring buffer of observed motion (NaN = not updated in that frame)."""

    def __init__(self, n, window, device):
        self.n, self.window = n, window
        self.xyz = torch.full((window, n), float("nan"), device=device, dtype=torch.float16)
        self.rot = torch.full((window, n), float("nan"), device=device, dtype=torch.float16)
        self.last_xyz = torch.full((n,), float("inf"), device=device)      # never observed -> active
        self.last_rot = torch.full((n,), float("inf"), device=device)
        self.ptr = 0

    def push(self, disp, angle, observed):
        nan = torch.tensor(float("nan"), device=disp.device)
        self.xyz[self.ptr] = torch.where(observed, disp.float(), nan).half()
        self.rot[self.ptr] = torch.where(observed, angle.float(), nan).half()
        self.last_xyz = torch.where(observed, disp.float(), self.last_xyz)
        self.last_rot = torch.where(observed, angle.float(), self.last_rot)
        self.ptr = (self.ptr + 1) % self.window

    def score(self):
        def nanmax(buf, last):
            b = buf.float()
            seen = ~torch.isnan(b)
            m = torch.where(seen, b, torch.full_like(b, -1.0)).amax(0)
            return torch.where(seen.any(0), m, last)
        return nanmax(self.xyz, self.last_xyz), nanmax(self.rot, self.last_rot)

    def state_dict(self):
        return dict(n=self.n, window=self.window, xyz=self.xyz.cpu(), rot=self.rot.cpu(),
                    last_xyz=self.last_xyz.cpu(), last_rot=self.last_rot.cpu(), ptr=self.ptr)

    @classmethod
    def from_state(cls, s, device):
        h = cls(s["n"], s["window"], device)
        h.xyz, h.rot = s["xyz"].to(device), s["rot"].to(device)
        h.last_xyz, h.last_rot = s["last_xyz"].to(device), s["last_rot"].to(device)
        h.ptr = s["ptr"]
        return h


def b_is_full_frame(k, warmup, recheck):
    """k = 1-based index of the streamed frame (frame000001 -> 1)."""
    return k <= warmup or (recheck > 0 and k % recheck == 0)


def select_by_motion(hist, tau_xyz, tau_rot_deg):
    sx, sr = hist.score()
    return (sx > tau_xyz) | (sr > tau_rot_deg)


def motion_quantiles(disp, angle, observed, qs=(0.5, 0.75, 0.9, 0.95, 0.99)):
    if observed.sum() == 0:
        return {}
    d = disp[observed].float()
    a = angle[observed].float()
    if d.numel() > 2_000_000:                      # torch.quantile input limit is 2^24; subsample for speed
        g = torch.Generator(device=d.device).manual_seed(0)
        pick = torch.randperm(d.numel(), generator=g, device=d.device)[:2_000_000]
        d, a = d[pick], a[pick]
    qt = torch.tensor(qs, device=d.device)
    dq, aq = torch.quantile(d, qt).tolist(), torch.quantile(a, qt).tolist()
    out = {f"disp_q{int(q * 100)}": v for q, v in zip(qs, dq)}
    out.update({f"rot_deg_q{int(q * 100)}": v for q, v in zip(qs, aq)})
    out["disp_mean"] = float(d.mean())
    return out



# ----------------------------------------------------------------------------------------- mode E / EB
def e_is_box_frame(k, n):
    """Recompute the adaptive box on streamed frame k (1-based): k == 1, 1+N, 1+2N, ..."""
    return n > 0 and (k - 1) % n == 0


def box_from_points(xyz, q_lo=0.01, q_hi=0.99, margin=0.1):
    """Percentile box of (M,3) points widened by margin * extent on each side. Returns (lo, hi) (3,)."""
    p = xyz.float()
    if p.shape[0] > 2_000_000:
        g = torch.Generator(device=p.device).manual_seed(0)
        p = p[torch.randperm(p.shape[0], generator=g, device=p.device)[:2_000_000]]
    qt = torch.tensor([q_lo, q_hi], device=p.device)
    q = torch.quantile(p, qt, dim=0)                 # (2,3)
    ext = (q[1] - q[0]).clamp_min(1e-6)
    pad = margin * ext + 1e-5                        # tiny pad: quantile round-off must not drop the extreme points
    return q[0] - pad, q[1] + pad


def in_box(xyz, lo, hi):
    return ((xyz >= lo) & (xyz <= hi)).all(dim=1)


class ChangeAccumulator:
    """Per-view OR of change masks over the frames since the last box recompute."""

    def __init__(self, tau=25.0, radius=48):
        self.tau, self.radius = tau, radius
        self.acc = {}
        self.pairs = 0

    def add(self, name, cur, prev):
        m = change_mask(cur, prev, self.tau, self.radius)
        a = self.acc.get(name)
        self.acc[name] = m if a is None else (a | m)

    def frame_done(self):
        self.pairs += 1

    def reset(self):
        self.acc, self.pairs = {}, 0


def select_box(xyz, accum, views, kappa=2, q_lo=0.01, q_hi=0.99, margin=0.1, min_points=1000):
    """views: iterable of (name, full_proj, world_view). Returns (lo, hi, stats) or (None, None, stats)."""
    count = torch.zeros(xyz.shape[0], device=xyz.device, dtype=torch.int32)
    frac = []
    for name, fp, wv in views:
        m = accum.acc.get(name)
        if m is None:
            continue
        frac.append(float(m.float().mean()))
        count += project_hits(xyz, fp, wv, m).int()
    cand = count >= kappa
    st = dict(box_candidates=int(cand.sum()), box_pairs=accum.pairs,
              box_mask_frac_mean=sum(frac) / max(len(frac), 1))
    if int(cand.sum()) < min_points:
        st["box_fallback"] = "too few candidates"
        return None, None, st
    lo, hi = box_from_points(xyz[cand], q_lo, q_hi, margin)
    st.update(box_lo=[round(float(v), 4) for v in lo], box_hi=[round(float(v), 4) for v in hi])
    return lo, hi, st
