"""Sim(3) alignment + quaternion / SH-band-1 rotation helpers (numpy)."""
import numpy as np


def umeyama(src, dst, w=None):
    """Find s,R,t minimizing sum w ||s R src + t - dst||^2. src,dst (N,3)."""
    if w is None:
        w = np.ones(len(src))
    w = w / w.sum()
    mu_s, mu_d = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = (w[:, None] * xd).T @ xs
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (w * (xs ** 2).sum(1)).sum()
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def umeyama_trimmed(src, dst, iters=4, keep=0.8):
    idx = np.arange(len(src))
    for _ in range(iters):
        s, R, t = umeyama(src[idx], dst[idx])
        res = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        idx = np.argsort(res)[: int(keep * len(src))]
    return s, R, t, res


def rotmat_to_quat_wxyz(R):
    m = R
    tr = np.trace(m)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S]
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    """Hamilton product, wxyz. a (4,) or (N,4), b (N,4)."""
    a = np.broadcast_to(a, b.shape)
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def rotate_sh1(sh1, R):
    """Rotate SH band 1. sh1 (N,3ch,3coef) in graphdeco basis:
    color += C1 * (-y*a0 + z*a1 - x*a2)  ==  C1 * dot(d, w),  w = (-a2, -a0, a1).
    Rotating the world by R: w' = R w."""
    a0, a1, a2 = sh1[..., 0], sh1[..., 1], sh1[..., 2]
    w = np.stack([-a2, -a0, a1], -1)          # (N,3ch,3xyz)
    w2 = w @ R.T
    out = np.empty_like(sh1)
    out[..., 0] = -w2[..., 1]
    out[..., 1] = w2[..., 2]
    out[..., 2] = -w2[..., 0]
    return out


def unproject(depth, K, w2c, z_depth=True):
    """depth (H,W), K 3x3, w2c 3x4/4x4 -> world points (H*W,3), pixel centers at +0.5."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
    rays = pix @ np.linalg.inv(K).T
    if not z_depth:
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    Xc = rays * depth.reshape(-1, 1)
    R, t = w2c[:3, :3], w2c[:3, 3]
    return (Xc - t) @ R          # R^T (Xc - t)
