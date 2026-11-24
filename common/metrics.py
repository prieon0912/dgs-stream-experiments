"""PSNR / SSIM (graphdeco implementations) / LPIPS-vgg (same backbone as 3DGStream's lpipsPyTorch)."""
import math

import numpy as np
import torch
import torch.nn.functional as F


def to_t(img, device="cuda"):
    """HxWx3 float numpy in [0,1] -> 3xHxW tensor."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().to(device)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float(20 * torch.log10(1.0 / torch.sqrt(mse)))


def _gauss(ws, sigma=1.5):
    g = torch.tensor([math.exp(-(x - ws // 2) ** 2 / (2 * sigma ** 2)) for x in range(ws)])
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].expand(3, 1, ws, ws).contiguous()


_W = {}


def ssim_t(a, b, ws=11):
    """differentiable, a,b (3,H,W)."""
    key = (a.device, ws)
    if key not in _W:
        _W[key] = _gauss(ws).to(a.device)
    w = _W[key]
    a, b = a[None], b[None]
    mu1 = F.conv2d(a, w, padding=ws // 2, groups=3)
    mu2 = F.conv2d(b, w, padding=ws // 2, groups=3)
    s11 = F.conv2d(a * a, w, padding=ws // 2, groups=3) - mu1 ** 2
    s22 = F.conv2d(b * b, w, padding=ws // 2, groups=3) - mu2 ** 2
    s12 = F.conv2d(a * b, w, padding=ws // 2, groups=3) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return m.mean()


def ssim(a, b):
    return float(ssim_t(a, b))


_LP = None


def lpips(a, b):
    global _LP
    if _LP is None:
        import lpips as _l
        _LP = _l.LPIPS(net="vgg", verbose=False).to(a.device).eval()
    with torch.no_grad():
        return float(_LP(a[None] * 2 - 1, b[None] * 2 - 1))


def all_metrics(pred, gt, do_lpips=True):
    pred = pred.clamp(0, 1)
    d = dict(psnr=psnr(pred, gt), ssim=ssim(pred, gt))
    if do_lpips:
        d["lpips"] = lpips(pred, gt)
    return d
