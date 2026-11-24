"""[any env] Write the train-only frame-0 scene (cam00 removed) used by Taming 3DGS and EDGS.

  python scripts/30_make_train_scene.py --frame_dir $FRAME0 --out $ALT_SCENE
See alt_init/colmap_subset.py for what is written. The source scene is only read.
"""
import argparse
import json

from alt_init.colmap_subset import make_train_scene
from common.colmap_io import load_cameras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--test_cam", default="cam00")
    args = ap.parse_args()
    info = make_train_scene(args.frame_dir, args.out, images=args.images, exclude=(args.test_cam,))
    # same views as our own refine/eval code uses for training
    ours = [c.name for c in load_cameras(args.frame_dir, images=args.images, exclude=(args.test_cam,))]
    check = [c.name for c in load_cameras(args.out, images=args.images)]
    if sorted(ours) != sorted(check) or sorted(check) != sorted(info["kept"]):
        raise SystemExit(f"view mismatch: ours={ours} subset={check}")
    print(json.dumps(dict(info, n_kept=len(info["kept"])), indent=1))


if __name__ == "__main__":
    main()
