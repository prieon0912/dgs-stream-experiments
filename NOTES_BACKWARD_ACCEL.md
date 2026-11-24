# 작업 노트: 역전파 가속 (SPEC_backward_accel.md) — 다음 에이전트용

브랜치 `backward-accel`. 상태: **Phase 0·1 코드 완료, GPU 검증 대기.** Phase 2 이후는 아직 시작하지 않음.
GPU 결과는 아직 하나도 없다(아래 숫자는 모두 코드를 읽어 확인한 사실이다).

## 1. 확인된 사실 (3DGStream `747ddfe`, 래스터라이저 `SJoJoK/3DGStreamRasterizer@39b62ce`)

| 항목 | 내용 | 근거 |
|---|---|---|
| 실행 설정 | `rotate_sh=false`, `sh_degree=1`, `white_background=false`, `eval=true`, `resolution=1`, `densification_interval=20`, `test_iterations=[150,250]`, `save_iterations=[150]`, `load_iteration=150` | `$SUITE/cfg_args.json` (08b가 프레임·NTC·test/save만 덮어써 `cfg_run.json` 생성). 서버의 실제 `cfg_run.json`은 직접 보지 못함 |
| 1단계 `requires_grad` | `load_ply`가 `_xyz/_features_*/_opacity/_scaling/_rotation`을 모두 `nn.Parameter(requires_grad=True)`로 만든다 → 1단계에서 이 기울기도 전부 계산됨(옵티마이저는 NTC만) | `scene/gaussian_model.py:321-326` |
| 2단계 densification | `adding_and_prune`은 `xyz_gradient_accum`(1단계 `add_densification_stats`의 2D 위치 기울기)만 읽음. `color_gradient_accum`은 쌓기만 하고 읽는 곳 없음 | `gaussian_model.py:543-547, 678-681` |
| 2단계 가우시안 선택 | AABB(NTC 경계) 안에서만. NTC는 밖의 가우시안을 움직이지 않으므로 "2단계 AABB 안" ⊆ "1단계 NTC 마스크" | `training_one_frame_s2_setup` |
| `rotate_sh=True`의 실제 효과 | **SH가 바뀌지 않는다.** `self._new_feature[mask][:,1:4] = ...`는 고급 인덱싱 복사본에 쓰는 것이라 원본에 반영 안 됨(업스트림 버그). 명세 배경의 "rotate_sh=True면 SH도 NTC 회전으로 바뀐다"는 코드상 사실이 아님 | `gaussian_model.py:683-703` |
| 래스터라이저 반환 순서 | `_C.rasterize_gaussians` → `(num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer)` (graphdeco는 depth 없음) | `rasterize_points.cu:116` |
| depth 출력 | `D = Σ z_view·α·T` — 알파로 **정규화하지 않은** 가중합 | `forward.cu:359,377` |
| 역전파 atomicAdd | `renderCUDA`에서 쌍마다 colors 3 + mean2D 2 + conic 3 + opacity 1 = 9회 (`bwd_depth=False`면 depth 분기 미실행) | `backward.cu:547-599` |
| gs 환경 | torch 2.0.1 + cu118, A6000 = sm_86 | `runpod/01_install_gs.sh` |
| 0_rendering2.png | 2단계 마지막 반복(250) test 카메라(cam00) 렌더, clamp 후 `torchvision.utils.save_image`. 250에서 저장 다음에 `adding_and_prune`이 한 번 더 돈다(20으로 나눠떨어짐) → 디스크의 added PLY = 렌더 시점 상태 | `train_frames.py` |

## 2. 다음 에이전트가 반드시 처리할 것

1. **R\* 래스터라이저 위치 미확인.** 레포에 `R*`/`M*` 관련 코드가 없다. Phase 2 전에 사용자에게 소스 위치를 묻고, 같은 `backward.cu`를 고쳤는지·이 명세의 변경과 충돌하는지 보고할 것. R\*가 v7 기본 구성이면 Phase 4 모든 조건에 깔아야 함.
2. **1단계 grad 플래그 — 명세대로 하면 틀린다.** `backward.cu:401`의 `computeColorFromSH`가 색 기울기를 `dL_dmeans`(→ NTC의 d_xyz)에 더한다. `sh_degree=1`이면 색이 시점 의존이므로 colors를 끄면 NTC 기울기가 원본과 달라진다. 1단계 flags는 `mean2D|conic|(sh_degree>0 또는 rotate_sh면 colors)`로 할 것. opacity는 1단계에서 학습되지 않으므로 끌 수 있음. 2.4 정확성 테스트에서 이 차이가 드러나야 정상.
3. **2단계 mask 순서:** `get_xyz = cat(_xyz, _added_xyz)` 순서. mask = [0]×기존 + [1]×추가, assert로 확인.
4. **기존 tile probe 결과 무효 열:** `results/tile_pairs_probe.csv`의 `visible_mean=-1`, `mean_radius=-1`, `pairs_per_visible` 빈칸은 probe 버그(반환 순서) 때문. `tools/tile_probe/probe.py`를 고쳤으니 Phase 0 재측정 결과로 대체할 것.
5. **fused-ssim 미검증:** 커밋 `a7c48d6`을 `runpod/06_install_fused_ssim.sh` 안에 고정. 업스트림 테스트 버전은 torch ≥ 2.3.1, gs 환경은 2.0.1. 빌드 실패 시 명세대로 Phase 1 중단·보고. 통과하면 `runpod/versions.lock.json`에 `fused_ssim_commit` 추가하고 `scripts/20_version_lock.py`의 `--key` choices도 늘릴 것. `--use_fast_math`로 빌드되므로 값 차이는 `tools/ssim_check.py`로만 판정.
6. **테스트 환경:** `tests/test_run_zipsplat_offline.py` 3개는 이 변경 전부터 실패(하위 프로세스가 `alt_init` import 실패, Windows). `PYTHONPATH=<repo>`로 pytest 실행.
7. **모드 B/C/EB와 grad skip:** 명세대로 AABB 기준 마스크만 쓸 것(선택 마스크를 쓰면 2단계 추가 위치가 바뀜).

## 3. 사용자 GPU 실행 대기 (Phase 0·1)

```bash
source $EXP/runpod/env.sh && conda activate gs && FINAL=$WS/final && cd $EXP
# Phase 0: init 4개 + 3DGS 15k + EDGS
python tools/tile_probe/probe.py --frame_dir $SCENE/frame000000 --out $WS/probe_phase0 \
  --glob "$FINAL/inits/*.ply" --ply 3dgs_15k=$BASE_INIT_PLY --glob "$ALT_RUNS/init/alt_s0_edgs_*/final.ply"
# Phase 1
bash $EXP/runpod/06_install_fused_ssim.sh
cd $REPO_GS && python $EXP/tools/ssim_check.py --scene $SCENE    # 20쌍: |값 차이| <= 1e-4, 기울기 상대 차이 <= 1e-3
```

## 4. 이번 브랜치에서 바꾼 파일
- `tools/tile_probe/probe.py`: radii를 반환 튜플에서 "이미지 다음의 첫 1-D 텐서"로 찾음(포크의 depth 건너뜀).
- `stream/train_frames_select.py`: `--fused_ssim {off,on}`(기본 off). on이면 `train_frames.ssim` 교체, 프레임 기록에 `fused_ssim: true`. off면 기존과 동일.
- `runpod/06_install_fused_ssim.sh`, `tools/ssim_check.py`, `tests/test_backward_hooks.py`(CPU 5개 통과).

## 5. 별도 작업: 키프레임 스트리밍(SPEC_keyframe_flowprior.md) — 로컬 전용
사용자 PC의 로컬 브랜치 `keyframe-wip`(커밋 `52dd62c`, **푸시 안 함**)에 Phase 1 미완성본이 있다:
`stream/keyframe.py`, 훅, audit/평가·동적 PSNR, `scripts/42~44`, `tests/test_keyframe.py`(34개 통과). 남은 것: `jobs/keyframe_s0/` 실행 스크립트, README, 보고.
이 작업은 이번 브랜치와 같은 `train_frames_select.py`를 고치므로 합칠 때 충돌 정리가 필요하다.
