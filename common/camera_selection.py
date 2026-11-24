"""Deterministic camera subsets. Geometry is COLMAP world / OpenCV camera (+Z forward)."""
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ("random", "position_fps", "direction_fps")


def validate_cameras(cameras):
    result, seen = [], set()
    for camera in cameras:
        name = camera["name"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"Empty or duplicate camera name: {name!r}")
        center = np.asarray(camera["center"], dtype=np.float64)
        forward = np.asarray(camera["forward"], dtype=np.float64)
        if center.shape != (3,) or forward.shape != (3,):
            raise ValueError(f"{name}: center and forward must each have 3 values")
        if not np.isfinite(center).all() or not np.isfinite(forward).all():
            raise ValueError(f"{name}: camera geometry must be finite")
        norm = np.linalg.norm(forward)
        if norm < 1e-12:
            raise ValueError(f"{name}: forward vector is zero")
        seen.add(name)
        result.append(dict(name=name, center=center.tolist(), forward=(forward / norm).tolist()))
    return sorted(result, key=lambda camera: camera["name"])


def load_geometry(frame_dir=None, cameras_json=None):
    if (frame_dir is None) == (cameras_json is None):
        raise ValueError("Supply exactly one of frame_dir or cameras_json")
    if cameras_json is not None:
        data = json.loads(Path(cameras_json).read_text(encoding="utf-8-sig"))
        cameras = data if isinstance(data, list) else data["cameras"]
        if isinstance(data, dict) and data.get("coordinate_system", "colmap_world") != "colmap_world":
            raise ValueError("Only colmap_world coordinates are supported")
        return validate_cameras(cameras)
    # No images, depth maps or GPU are needed: only COLMAP image extrinsics.
    from common.colmap_io import read_images_bin, qvec2rotmat
    root = Path(frame_dir)
    choices = (root / "sparse/0/images.bin", root / "sparse/images.bin")
    path = next((path for path in choices if path.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No COLMAP images.bin in {root}/sparse[/0]")
    cameras = []
    for camera in read_images_bin(path).values():
        rotation = qvec2rotmat(camera["qvec"])
        cameras.append(dict(name=Path(camera["name"]).stem,
                            center=(-rotation.T @ camera["tvec"]).tolist(),
                            forward=(rotation.T @ np.array([0., 0., 1.])).tolist()))
    return validate_cameras(cameras)


def selection_order(cameras, method, seed=0, anchor=None):
    """All methods share one reference camera; random samples the remaining cameras.

    A single ordering gives nested 4/8/12/18 subsets within each method.
    Sorted input and argmax's first-index tie break make FPS reproducible.
    """
    cameras = validate_cameras(cameras)
    if not cameras or method not in METHODS:
        raise ValueError("No eligible cameras, or unknown method")
    names = [camera["name"] for camera in cameras]
    anchor = names[0] if anchor is None else anchor
    if anchor not in names:
        raise ValueError(f"Anchor camera {anchor!r} is absent or excluded")
    first = names.index(anchor)
    if method == "random":
        others = [index for index in range(len(names)) if index != first]
        indices = [first] + np.random.default_rng(seed).permutation(others).tolist()
    else:
        values = np.asarray([camera["center" if method == "position_fps" else "forward"]
                             for camera in cameras], dtype=np.float64)
        if method == "position_fps":
            distances = np.sum((values[:, None] - values[None, :]) ** 2, axis=2)
        else:
            # Squared chord distance is monotonic with angle between unit vectors.
            distances = np.maximum(0., 2. - 2. * np.clip(values @ values.T, -1., 1.))
        indices = [first]
        nearest = distances[first].copy()
        while len(indices) < len(names):
            nearest[indices] = -np.inf
            chosen = int(np.argmax(nearest))
            indices.append(chosen)
            nearest = np.minimum(nearest, distances[chosen])
    return [names[index] for index in indices]


def validate_view_names(names, excluded=(), available=None):
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("camera_names must be a non-empty list of camera names")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate input camera names")
    leaked = sorted(set(names).intersection(excluded))
    if leaked:
        raise ValueError(f"Held-out camera in DA3 input: {leaked}")
    if available is not None:
        missing = sorted(set(names).difference(available))
        if missing:
            raise ValueError(f"Requested cameras not found: {missing}")
    return names


def read_views_file(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return validate_view_names(data), {}
    if not isinstance(data, dict) or "camera_names" not in data:
        raise ValueError("Views JSON must be a list or an object with camera_names")
    validate_view_names(data["camera_names"], data.get("excluded_cameras", ()))
    return data["camera_names"], data


def geometry_summary(cameras):
    centers = np.asarray([camera["center"] for camera in cameras], dtype=np.float64)
    directions = np.asarray([camera["forward"] for camera in cameras], dtype=np.float64)
    upper = np.triu_indices(len(cameras), k=1)
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)[upper]
    angles = np.degrees(np.arccos(np.clip(directions @ directions.T, -1., 1.)))[upper]
    return dict(min_pair_center_distance=float(distances.min()) if len(distances) else None,
                min_pair_forward_angle_deg=float(angles.min()) if len(angles) else None,
                max_pair_forward_angle_deg=float(angles.max()) if len(angles) else None,
                center_min=centers.min(axis=0).tolist(), center_max=centers.max(axis=0).tolist())


def make_plan(cameras, counts=(4, 8, 12, 18), seed=0, excluded=("cam00",), anchor=None):
    all_cameras = validate_cameras(cameras)
    excluded = sorted(set(excluded))
    eligible = [camera for camera in all_cameras if camera["name"] not in excluded]
    counts = list(counts)
    if not counts or len(counts) != len(set(counts)) or any(type(count) is not int or count < 1 for count in counts):
        raise ValueError("counts must be unique positive integers")
    if len(eligible) < max(counts):
        raise ValueError(f"Need {max(counts)} eligible cameras; found {len(eligible)} after exclusions")
    anchor = eligible[0]["name"] if anchor is None else anchor
    by_name = {camera["name"]: camera for camera in eligible}
    records, known_sets = [], {}
    for method in METHODS:
        order = selection_order(eligible, method, seed, anchor)
        for count in counts:
            priority = order[:count]
            names = [anchor] + sorted(name for name in priority if name != anchor)
            condition = f"{method}_{count:02d}"
            digest = hashlib.sha256(json.dumps(sorted(names)).encode()).hexdigest()
            records.append(dict(condition_id=condition, method=method, n_views=count, seed=seed,
                                anchor=anchor, camera_names=names, selection_order=priority,
                                excluded_cameras=excluded, view_set_sha256=digest,
                                same_view_set_as=known_sets.get(digest),
                                geometry=geometry_summary([by_name[name] for name in names])))
            known_sets.setdefault(digest, condition)
    return dict(schema_version=1, coordinate_system="colmap_world", counts=counts, seed=seed,
                anchor=anchor, excluded_cameras=excluded,
                eligible_cameras=[camera["name"] for camera in eligible],
                conditions=records,
                notes=["All methods include the same anchor as DA3's first/reference view.",
                       "Random means seeded sampling without replacement of non-anchor cameras.",
                       "Subsets are nested within a method; camera geometry alone does not measure visibility or image quality.",
                       "Equal view sets across methods are reported, not silently treated as distinct evidence."])


def save_json_checked(path, data):
    path = Path(path)
    if path.exists() and json.loads(path.read_text(encoding="utf-8-sig")) != data:
        raise FileExistsError(f"Different configuration already exists: {path}; choose a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
