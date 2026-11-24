# Paths + pinned commits of the alternative first-frame initializers (source after runpod/env.sh).
# Each method lives in its OWN conda env, so its CUDA extensions (several are called
# diff_gaussian_rasterization / simple_knn) never replace the 3DGStream ones in env "gs".
export ALT=$WS/alt_init                         # external repos
export TAMING_ROOT=$ALT/taming-3dgs
export EDGS_ROOT=$ALT/EDGS
export ZIPSPLAT_ROOT=$ALT/ZipSplat
export ALT_RUNS=$RUNS/alt_init                  # all outputs of this experiment
export ALT_SCENE=$ALT_RUNS/scene_train          # frame 0 without cam00 (30_make_train_scene.py)
export TRITON_CACHE_DIR=$WS/.cache/triton       # ZipSplat kernels: compiled once, kept across pod restarts

# commits checked on 2026-11-24 (HEAD of the default branch)
export TAMING_COMMIT=fd0f7d9edfe135eb4eefd3be82ee56dada7f2a16
export EDGS_COMMIT=f90b022445fc88368f75e66e8fb34aea88372cac       # pins RoMa 370117431ff.., gaussian-splatting 54c035f7..
export ZIPSPLAT_COMMIT=73ab63f048fcc15ee54ada4ef9cdd79b925611a8
