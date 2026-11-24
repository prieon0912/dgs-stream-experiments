# FF init vs Optimization init — 3DGStream / flame_steak (RunPod 런북)

카메라 선택 12개 조건과 수동 AABB 하나를 공통 적용하는 추가 기능은
[README_CAMERA_AABB.md](README_CAMERA_AABB.md)를 먼저 확인하세요.
기존 3개 + 추가 12개를 합쳐 총 15개 조건이며, AABB 교차 실험을 추가하지 않습니다.

## 측정 항목 (최종 표 한 행 = 파이프라인 하나)
| 구분 | 항목 | 어디서 나오나 |
|---|---|---|
| init | 생성 시간, 단계별 내역 | `<init>_timing.json` (00 / 02+03 / 05 / 06이 기록하고 다음 단계로 누적) |
| init | 가우시안 수, 용량(MB), cam00 PSNR/SSIM/LPIPS | `09` frame 0 행 (`04`는 빠른 사전 점검용) |
| 스트리밍 | 프레임별 PSNR/SSIM/LPIPS | `per_frame.csv` |
| 스트리밍 | 프레임별 Preparation / Stage1 / Stage2 / wall 시간 | `train.log` 파싱 |
| 스트리밍 | 전체 평균: **1..N-1**과 **0..N-1**(논문 방식, init 포함), 앞 30 → 뒤 30 drift | `run_summary.json` |
| 스트리밍 | NTC warm-up 시간, 총 학습 시간, end-to-end 시간 | `run_summary.json` |
| 스트리밍 | 저장 용량 (init + Σ(NTC + 추가 가우시안))/N, 최대 GPU 메모리 | `run_summary.json` |
| 비교 | 표, 프레임별 그래프 4개 패널, init 시간 vs 품질 산점도 | `10_report.py` |

## 측정 프로토콜 (주의할 점)
1. **COLMAP 시간은 모든 init에서 제외합니다.** 두 파이프라인 모두 같은 COLMAP 포즈를 전제로 하기 때문입니다. 기준 init은 COLMAP 포인트도 쓰는데, 이 비대칭은 결과에 한 줄로 적습니다.
2. **가중치 로딩은 제외하고 따로 기록합니다** (DA3 `model_load_sec`). 기준 init 시간은 `train.py`의 wall clock이라서 시작 오버헤드 약 5~10초와 이미지 로딩이 포함됩니다(json `notes`에 기재). DA3 시간도 이미지 로딩과 전처리를 포함하므로 조건은 거의 같습니다.
3. **시간을 재는 단계는 GPU를 혼자 써야 합니다.** 각 스크립트가 시작할 때 GPU 사용률을 확인해 `gpu_exclusive`를 기록합니다. False이면 표에서 `time OK = False`로 표시되고, 그 시간은 보고하면 안 됩니다. 그래서 아래 순서는 **init 생성을 전부 먼저 하고, 스트리밍 큐는 그 다음**에 돌리도록 되어 있습니다.
4. **같은 GPU 종류로 측정합니다.** Pod를 바꾸면 처음부터 다시 잽니다(`gpu` 필드에 GPU 이름이 기록됨).
5. **DA3 시간은 첫 호출(cold) 기준입니다.** cuDNN autotune 등이 포함되며, 기준 init의 학습도 cold이므로 조건이 같습니다.
6. **refine 시간에서 중간 test 평가 시간은 뺍니다.** 그래서 순수 최적화 시간만 남습니다.
7. **스트리밍 시간은 Stage 1+2(순수 학습)와 wall(이미지 로딩 포함)을 둘 다 보고합니다.** 디스크 속도에 따라 Preparation 값이 달라지기 때문입니다.
8. **지표 정의:** PSNR은 float 값을 씁니다(frame ≥1은 3DGStream이 로그에 남긴 값). SSIM과 LPIPS는 모든 프레임을 8-bit 이미지로 계산합니다(frame 0도 8-bit로 양자화). LPIPS를 `--lpips_every 10`으로 계산하면 평균은 10프레임 간격 샘플의 평균이라고 적습니다.
9. **비교 기준은 우리가 직접 학습한 `baseline_own`입니다.** 시간과 지표가 같은 모델에서 나와야 하기 때문입니다. suite init(`base_official`)은 3DGStream 재현 확인용입니다.
10. **NTC warm-up은 모든 init에 동일하게 다시 합니다**(같은 AABB). warm-up 시간은 init 시간과 따로 표시하고, end-to-end 시간에는 포함합니다.
11. **init 가우시안 수가 다르면 스트리밍 시간도 달라집니다.** DA3에서 top-K로 자른 경우 K값과 원래 가우시안 수를 표에 적습니다.

## 0. Pod 설정
| 항목 | 권장 |
|---|---|
| GPU | A6000 48GB / A100 80GB / L40S (24GB는 DA3 가우시안 수를 줄여야 함). **처음부터 끝까지 같은 GPU** |
| Template | **devel** 이미지: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` |
| Volume (/workspace) | **≥ 80GB** |
| 종류 | On-demand 권장 (Spot이어도 스트리밍은 이어서 진행됨) |

- /workspace만 Pod 재시작 후에도 남습니다. 재시작할 때마다 `00_setup_pod.sh`를 실행하세요. 오래 걸리는 작업은 tmux 안에서 돌립니다.
- 이 폴더를 `/workspace/dgs_exp`로 업로드하고, 터미널마다 `source runpod/env.sh`를 실행합니다.

## 1. 세팅 (0~1.5h)
```bash
cd /workspace/dgs_exp && source runpod/env.sh
bash runpod/00_setup_pod.sh
tmux new -s setup
#  창 0: bash runpod/01_install_gs.sh      (~20분)
#  창 1: bash runpod/02_install_da3.sh     (~10분 + 가중치 다운로드)
#  창 2: (01 끝난 뒤) bash runpod/03_get_data.sh   (~30-60분, CPU, 재실행하면 이어서 진행)
```
시간이 부족하면 `export N_FRAMES=150`으로 모든 init에 똑같이 적용합니다.

## 2. init 생성 — 시간 측정 구간, GPU 단독 (~30분)
**큐를 돌리기 전에** 순서대로 실행합니다. 각 단계가 `_timing.json`을 남깁니다.
```bash
conda activate gs
# (a) Optimization init: 3DGS 15k, SH1, cam00 제외  (~10-15분)
bash scripts/00_train_baseline.sh 15000 baseline_own

# (b) FF init: DA3 추론(시간 측정) -> 변환(시간 측정)
conda activate da3
python scripts/02_run_da3.py --frame_dir $FRAME0 --out $RUNS/da3/da3_raw.pt
conda activate gs
python scripts/03_convert_da3.py --raw $RUNS/da3/da3_raw.pt --out $RUNS/inits/da3_full.ply
python scripts/06_subsample.py --ply $RUNS/inits/da3_full.ply --out $RUNS/inits/da3_raw.ply --topk 1000000

# (c) DA3 + 1k refine, 개수 맞춘 버전
python scripts/05_refine.py --ply $RUNS/inits/da3_raw.ply --out $RUNS/inits/da3_1k.ply --frame_dir $FRAME0 --iters 1000
python scripts/06_subsample.py --ply $RUNS/inits/da3_full.ply --out $RUNS/inits/da3_count.ply --match_ref $RUNS/inits/baseline_own.ply

# 사전 점검: 정렬, AABB, 정적 지표, init 시간을 한눈에
python scripts/04_check_and_eval.py --frame_dir $FRAME0 --out $RUNS/static --train \
  --ply baseline_own=$RUNS/inits/baseline_own.ply --ply suite=$BASE_INIT_PLY \
  --ply da3_raw=$RUNS/inits/da3_raw.ply --ply da3_1k=$RUNS/inits/da3_1k.ply --ply da3_count=$RUNS/inits/da3_count.ply
grep -h '"init_time_sec"\|gpu_exclusive' $RUNS/inits/*_timing.json
```
**판단 기준:** `$RUNS/static/da3_raw_cam00_panel.png`의 4번째 칸(50% 블렌드)에서 윤곽이 이중으로 보이거나, 03의 잔차가 2%를 넘으면 정렬에 실패한 것입니다.
1. `--ray_depth`로 다시 시도합니다.
2. 그래도 안 되면 플랜 B로 갑니다.
```bash
python scripts/03b_depth_to_init.py --raw $RUNS/da3/da3_raw.pt --out $RUNS/inits/depth_pts.ply
python scripts/05_refine.py --ply $RUNS/inits/depth_pts.ply --out $RUNS/inits/depth_300.ply --frame_dir $FRAME0 --iters 300
python scripts/05_refine.py --ply $RUNS/inits/depth_pts.ply --out $RUNS/inits/depth_1k.ply  --frame_dir $FRAME0 --iters 1000
```

## 3. 스트리밍 (init당 약 1시간, 큐로 연속 실행)
```bash
cat > $RUNS/queue.txt <<EOF
baseline_own  $RUNS/inits/baseline_own.ply
da3_raw       $RUNS/inits/da3_raw.ply
da3_1k        $RUNS/inits/da3_1k.ply
da3_count     $RUNS/inits/da3_count.ply
base_official $BASE_INIT_PLY official
EOF
bash scripts/run_queue.sh $RUNS/queue.txt       # tmux 세션 'q'
tail -f $RUNS/stream/baseline_own/train.log
```
- 큐가 도는 동안 **같은 GPU에서 다른 작업을 하지 마세요.** 스트리밍 시간이 오염됩니다.
- OOM이 나면 `06_subsample.py --topk 500000`으로 줄인 init을 새 이름으로 넣고, 그 사실을 기록합니다.
- Pod가 재시작되면 `00_setup_pod.sh`를 실행한 뒤 같은 큐 명령을 다시 실행합니다(끝난 프레임부터 이어서 진행).

## 4. 지표 + 보고서
```bash
python scripts/09_eval_stream.py --runs baseline_own da3_raw da3_1k da3_count base_official --lpips_every 10
python scripts/10_report.py --out $RUNS/report \
  --runs 'baseline_own:3DGS 15k' 'da3_raw:DA3' 'da3_1k:DA3+1k' 'da3_count:DA3 (#G matched)'
```
`$RUNS/report/` 결과물:
- `summary.md` / `summary.csv`: 비교 표와 해석 초안
- `fig_per_frame.png`: PSNR/SSIM/LPIPS/학습 시간 4개 패널
- `fig_time_quality.png`: init 시간 vs 품질 산점도

`--runs`의 첫 번째 항목이 비교 기준입니다.

## 결과에 반드시 적을 것
- GPU 이름, 그리고 모든 시간 측정이 `time OK = True`인지
- 기준 init: 3DGS 15k, SH1, densification 기본값, images_2, cam00 제외
- DA3: 모델(`DA3-GIANT-1.1`), process_res 504, 입력 cam01~20, GT 포즈로 조건 부여, Sim(3) 잔차, SH1 사용, top-K 값
- 시간 정의: COLMAP과 가중치 로딩 제외, NTC warm-up은 별도 표시
- 플랜 B로 갔다면 그 사실
- `base_official`과 `baseline_own`의 차이(suite init으로 재현했을 때와의 차이)

## 파일 구성
| 파일 | env | 역할 |
|---|---|---|
| `runpod/env.sh`, `00~03` | — | 경로, 캐시, apt, conda, 설치, 데이터 |
| `common/timing.py` | any | init 시간 기록과 누적, GPU 단독 사용 여부 확인 |
| `scripts/00_train_baseline.sh` | gs | Optimization init을 직접 학습하고 시간 측정 |
| `scripts/01_prep_dataset.py` | gs | 동영상 → 3DGStream 데이터 구조 (재개 가능) |
| `scripts/02_run_da3.py` | da3 | DA3 추론과 시간 측정 → 원본 .pt |
| `scripts/03_convert_da3.py` | gs | 픽셀 대응 Sim(3) 정렬 + SH1 회전 → PLY (변환 시간 누적) |
| `scripts/03b_depth_to_init.py` | gs | 플랜 B: depth 포인트 → 초기 가우시안 |
| `scripts/04_check_and_eval.py` | gs | 사전 점검: 개수, 용량, AABB, 정적 지표, 겹쳐 보기 이미지 |
| `scripts/05_refine.py` | gs | densification 없는 짧은 학습 (시간 누적) |
| `scripts/06_subsample.py` | any | top-K 또는 기준 개수에 맞추기 (시간 누적) |
| `scripts/07_ntc_warmup.py` | gs | NTC warm-up과 시간 측정 |
| `scripts/08_run_stream.sh` | gs | 스트리밍 1회 (재개 가능, GPU 단독 여부와 메모리 기록) |
| `scripts/run_queue.sh` | gs | tmux 큐 |
| `scripts/09_eval_stream.py` | gs | 프레임별 지표, 시간, 용량 → `per_frame.csv`, `run_summary.json` |
| `scripts/10_report.py` | gs | 비교 표, 그래프 2장, 해석 초안 |
