"""Train-only copy of the frame-0 COLMAP scene (cam00 removed), for external 3DGS code bases.

Why: Taming 3DGS (llffhold=8 with --eval) and EDGS (its own "every 8th camera" split) would
otherwise hold out cam01/cam08/cam16-style views instead of cam00. Giving them a scene that
only contains the 20 training views (cam01..cam20), and running them WITHOUT their own eval split, makes
every method see exactly the same images as our refine (05_refine.py excludes cam00).

What is written (nothing in the source scene is modified):
  <out>/<images>/camXX.png      symlinks to the source images (cam00 skipped)
  <out>/sparse/0/cameras.bin    copied unchanged (an unused cam00 intrinsics entry is harmless)
  <out>/sparse/0/points3D.bin   copied unchanged (readers ignore the tracks)
  <out>/sparse/0/images.bin     rewritten without cam00, with no 2D points
  <out>/subset.json             which views were kept / dropped
"""
import json
import os
import shutil
import struct

from common.colmap_io import read_images_bin


def _sparse_dir(frame_dir):
    sp = os.path.join(frame_dir, "sparse", "0")
    return sp if os.path.isdir(sp) else os.path.join(frame_dir, "sparse")


def write_images_bin(path, images):
    """images: iterable of dicts with id, qvec, tvec, camera_id, name (COLMAP binary layout)."""
    images = list(images)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images:
            f.write(struct.pack("<i", int(im["id"])))
            f.write(struct.pack("<4d", *[float(x) for x in im["qvec"]]))
            f.write(struct.pack("<3d", *[float(x) for x in im["tvec"]]))
            f.write(struct.pack("<i", int(im["camera_id"])))
            f.write(im["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))          # no 2D points


def make_train_scene(frame_dir, out_dir, images="images_2", exclude=("cam00",)):
    sp = _sparse_dir(frame_dir)
    extr = read_images_bin(os.path.join(sp, "images.bin"))
    keep, dropped = [], []
    for key, im in sorted(extr.items(), key=lambda kv: kv[1]["name"]):
        stem = os.path.splitext(im["name"])[0]
        (dropped if stem in exclude else keep).append(dict(im, id=im.get("id", key)))
    if not keep:
        raise RuntimeError("no views left after excluding " + ",".join(exclude))
    missing = [e for e in exclude if e not in {os.path.splitext(d["name"])[0] for d in dropped}]
    if missing:
        raise RuntimeError(f"excluded views not found in {sp}: {missing}")

    out_sp = os.path.join(out_dir, "sparse", "0")
    out_img = os.path.join(out_dir, images)
    os.makedirs(out_sp, exist_ok=True)
    os.makedirs(out_img, exist_ok=True)
    for name in ("cameras.bin", "points3D.bin"):
        shutil.copyfile(os.path.join(sp, name), os.path.join(out_sp, name))
    write_images_bin(os.path.join(out_sp, "images.bin"), keep)
    for im in keep:
        src = os.path.abspath(os.path.join(frame_dir, images, im["name"]))
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        dst = os.path.join(out_img, im["name"])
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    info = dict(source=os.path.abspath(frame_dir), images=images,
                kept=[os.path.splitext(i["name"])[0] for i in keep],
                dropped=[os.path.splitext(i["name"])[0] for i in dropped])
    with open(os.path.join(out_dir, "subset.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info
