"""Shared AABB configuration, point diagnostics, and safe streaming-run identity."""
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_MIN = [-20., -15., 5.]
DEFAULT_MAX = [15., 10., 23.]


def canonical_aabb(data):
    if data.get("coordinate_system", "colmap_world") != "colmap_world":
        raise ValueError("AABB must use colmap_world coordinates, after DA3 alignment")
    lo, hi = np.asarray(data["min"], dtype=np.float32), np.asarray(data["max"], dtype=np.float32)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("AABB min/max must be finite 3-vectors representable in float32")
    if not np.all(hi > lo) or not np.isfinite(hi - lo).all():
        raise ValueError("AABB max must exceed min in every axis")
    return dict(schema_version=1, coordinate_system="colmap_world", min=lo.tolist(), max=hi.tolist())


def aabb_id(data):
    canonical = canonical_aabb(data)
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def load_aabb(path=None):
    data = (json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else
            dict(min=DEFAULT_MIN, max=DEFAULT_MAX, method="original_fixed"))
    result = dict(data)
    result.update(canonical_aabb(data))
    result["aabb_id"] = aabb_id(result)
    return result


def save_aabb(path, data):
    result = dict(data)
    result.update(canonical_aabb(data))
    result["aabb_id"] = aabb_id(result)
    path = Path(path)
    if path.exists() and json.loads(path.read_text(encoding="utf-8-sig")) != result:
        raise FileExistsError(f"AABB file already has different settings: {path}; use a new filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def load_xyz(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        xyz = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        xyz = data["xyz"] if isinstance(data, dict) else data
    elif path.suffix.lower() == ".ply":
        from plyfile import PlyData
        vertex = PlyData.read(path)["vertex"]
        xyz = np.column_stack([vertex[key] for key in ("x", "y", "z")])
    else:
        raise ValueError("Point file must be .ply, .npy or .json")
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError(f"Expected non-empty (N, 3) points: {path}")
    return xyz


def inside_mask(xyz, aabb):
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Expected (N, 3) points")
    box = canonical_aabb(aabb)
    return np.isfinite(xyz).all(axis=1) & ((xyz >= box["min"]) & (xyz <= box["max"])).all(axis=1)


def point_stats(xyz, aabb):
    xyz = np.asarray(xyz)
    inside = inside_mask(xyz, aabb)
    finite = np.isfinite(xyz).all(axis=1)
    good = xyz[finite]
    return dict(n_total=int(len(xyz)), n_finite=int(finite.sum()), n_nonfinite=int((~finite).sum()),
                n_inside=int(inside.sum()), n_outside=int(len(xyz) - inside.sum()),
                inside_fraction=float(inside.mean()) if len(xyz) else None,
                xyz_min=good.min(axis=0).tolist() if len(good) else None,
                xyz_max=good.max(axis=0).tolist() if len(good) else None,
                xyz_p01=np.quantile(good, .01, axis=0).tolist() if len(good) else None,
                xyz_p99=np.quantile(good, .99, axis=0).tolist() if len(good) else None)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_same_bounds(actual, expected):
    # Both sides canonicalize to float32, matching the buffers saved by warm-up.
    if aabb_id(actual) != aabb_id(expected):
        raise ValueError("NTC/AABB mismatch: use the correct checkpoint or a new run name and re-warm-up")


def inspect_ntc(path, expected=None):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    bounds = dict(min=state["xyz_bound_min"].detach().cpu().tolist(),
                  max=state["xyz_bound_max"].detach().cpu().tolist(), method="ntc_checkpoint",
                  ntc_path=str(Path(path).resolve()))
    bounds.update(canonical_aabb(bounds))
    bounds["aabb_id"] = aabb_id(bounds)
    if expected is not None:
        assert_same_bounds(bounds, expected)
    return bounds


def bind_run(run_dir, aabb, ply, ntc_config, ntc_mode, environment=None):
    """Refuse resume with changed inputs/bounds, or an untracked legacy checkpoint."""
    root = Path(run_dir)
    metadata = dict(schema_version=1, aabb=canonical_aabb(aabb), aabb_id=aabb_id(aabb),
                    init_sha256=file_sha256(ply), ntc_config_sha256=file_sha256(ntc_config),
                    ntc_mode=ntc_mode)
    if environment is not None:
        from common.provenance import environment_fingerprint
        metadata["environment_fingerprint"] = environment_fingerprint(environment)
    path = root / "run_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Existing run uses different PLY/AABB/NTC config/mode. Choose a new run name.")
    else:
        if (root / "ntc.pth").exists() or (root / "out").exists():
            raise ValueError("Existing run has no run_contract.json. Use a new run name to avoid mixing old results.")
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    save_aabb(root / "aabb.json", metadata["aabb"])
    return metadata
