"""Fixed-resolution center occupancy diagnostics; these do NOT prove redundant geometry."""
from pathlib import Path

import numpy as np

from common.aabb import canonical_aabb, inside_mask


def sources_path(ply):
    return Path(ply).with_name(Path(ply).stem + "_sources.npz")


def occupancy(xyz, aabb, voxel_size=None, source_views=None):
    box = canonical_aabb(aabb)
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Occupancy requires non-empty, finite (N, 3) centers")
    voxel_size = float(voxel_size or (np.max(np.subtract(box["max"], box["min"])) / 128))
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    cells = np.floor((xyz - box["min"]) / voxel_size).astype(np.int64)
    unique, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    result = dict(voxel_size=voxel_size, n_points=int(len(xyz)), occupied_voxels=int(len(unique)),
                  mean_centers_per_occupied_voxel=float(counts.mean()),
                  p95_centers_per_occupied_voxel=float(np.quantile(counts, .95)),
                  fraction_points_in_multi_center_voxels=float((counts[inverse] > 1).mean()),
                  n_centers_inside_aabb=int(inside_mask(xyz, box).sum()),
                  multi_view_voxels=None, fraction_points_in_multi_view_voxels=None,
                  diagnostic_scope="center occupancy proxy; not Gaussian support intersection or a redundancy verdict")
    if source_views is not None:
        source_views = np.asarray(source_views)
        if source_views.shape != (len(xyz),) or not np.issubdtype(source_views.dtype, np.integer):
            raise ValueError("Source-view IDs must be an integer vector matching Gaussian order")
        if source_views.min() < 0 or source_views.max() >= 64:
            raise ValueError("Source-view IDs must be in [0, 63]")
        masks = np.zeros(len(unique), dtype=np.uint64)
        np.bitwise_or.at(masks, inverse, np.left_shift(np.uint64(1), source_views.astype(np.uint64)))
        multiple = (masks & (masks - np.uint64(1))) != 0
        result.update(multi_view_voxels=int(multiple.sum()),
                      fraction_points_in_multi_view_voxels=float(multiple[inverse].mean()))
    return result
