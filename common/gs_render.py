"""Standalone 3DGS renderer using the installed diff_gaussian_rasterization (3DGStream fork in env "gs").
Same projection convention as graphdeco (FoV from focal, centered principal point), so numbers are
comparable with 3DGStream's own test PSNR."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from common.ply_io import load_ply, sh_degree_of


def _proj(znear, zfar, fovx, fovy):
    tx, ty = math.tan(fovx / 2), math.tan(fovy / 2)
    top, right = ty * znear, tx * znear
    P = torch.zeros(4, 4)
    P[0, 0] = 2 * znear / (2 * right)
    P[1, 1] = 2 * znear / (2 * top)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class RCam:
    def __init__(self, cam, device="cuda"):
        self.name, self.W, self.H = cam.name, cam.width, cam.height
        self.fovx, self.fovy = cam.fovx, cam.fovy
        self.wv = torch.tensor(cam.w2c, dtype=torch.float32).T.contiguous().to(device)
        P = _proj(0.01, 100.0, self.fovx, self.fovy).T.to(device)
        self.full = (self.wv.unsqueeze(0).bmm(P.unsqueeze(0))).squeeze(0)
        self.campos = self.wv.inverse()[3, :3]


class GS(torch.nn.Module):
    """Trainable gaussian container (graphdeco layout)."""

    def __init__(self, d, device="cuda"):
        super().__init__()
        n = len(d["xyz"])
        self.sh_degree = sh_degree_of(d)
        k = (self.sh_degree + 1) ** 2 - 1
        t = lambda a: torch.nn.Parameter(torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device))
        self.xyz = t(d["xyz"])
        self.f_dc = t(d["f_dc"].reshape(n, 3, 1).transpose(0, 2, 1))          # (N,1,3)
        self.f_rest = t(d["f_rest"].reshape(n, 3, k).transpose(0, 2, 1))      # (N,K-1,3)
        self.opacity = t(d["opacity"].reshape(n, 1))
        self.scale = t(d["scale"])
        self.rot = t(d["rot"])

    @classmethod
    def from_ply(cls, path, device="cuda"):
        return cls(load_ply(path), device)

    def to_dict(self):
        n = self.xyz.shape[0]
        c = lambda x: x.detach().cpu().numpy()
        return dict(xyz=c(self.xyz), f_dc=c(self.f_dc).transpose(0, 2, 1).reshape(n, 3),
                    f_rest=c(self.f_rest).transpose(0, 2, 1).reshape(n, -1),
                    opacity=c(self.opacity).reshape(n), scale=c(self.scale), rot=c(self.rot))


_FIELDS = GaussianRasterizationSettings._fields


def render(gs, rc, bg=(0.0, 0.0, 0.0), sh_degree=None):
    sh_degree = gs.sh_degree if sh_degree is None else sh_degree
    bg_t = torch.tensor(bg, dtype=torch.float32, device=gs.xyz.device)
    kw = dict(image_height=rc.H, image_width=rc.W, tanfovx=math.tan(rc.fovx / 2), tanfovy=math.tan(rc.fovy / 2),
              bg=bg_t, scale_modifier=1.0, viewmatrix=rc.wv, projmatrix=rc.full, sh_degree=sh_degree,
              campos=rc.campos, prefiltered=False, debug=False, bwd_depth=False, antialiasing=False)
    settings = GaussianRasterizationSettings(**{k: v for k, v in kw.items() if k in _FIELDS})
    means2D = torch.zeros_like(gs.xyz, requires_grad=True)
    out = GaussianRasterizer(raster_settings=settings)(
        means3D=gs.xyz, means2D=means2D,
        shs=torch.cat([gs.f_dc, gs.f_rest], 1).contiguous(), colors_precomp=None,
        opacities=torch.sigmoid(gs.opacity), scales=torch.exp(gs.scale),
        rotations=F.normalize(gs.rot, dim=-1), cov3D_precomp=None)
    img = out[0] if isinstance(out, (tuple, list)) else out
    return img  # (3,H,W), not clamped
