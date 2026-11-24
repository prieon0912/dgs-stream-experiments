# 실행 및 결과 CSV

## 고정된 실험 설계

총 15개 조건이다. 기준 3개(3DGS 15k, 전체 학습 뷰 DA3, 전체 학습 뷰 DA3+1k)와 카메라 선택 12개를 비교한다.
모든 조건에서 같은 AABB와 GPU를 사용한다. 추가 12개는 DA3 초기화 입력 카메라만 달라지고,
1,000회 보정과 스트리밍에는 동일한 20개 학습 카메라를 사용한다. cam00은 평가용으로 제외한다.

모든 DA3 조건은 같은 변환 필터를 사용한다: border=0.03125, depth_prune=1.0, min_opacity=0.005, SH1 유지.
현재 실행 계획에는 `--max-gaussians 1000000`을 명시했다. 변환 후 불투명도가 높은 가우시안을
최대 100만 개 남기고 보정 학습한다. 이미 100만 개보다 적으면 그대로 사용한다.
조건 하나만 개수를 줄여서 기존 결과와 섞지 않는다.
3DGS는 자체 학습으로 생성한 개수를 그대로 사용한다.

카메라 수를 줄였다고 모든 단계의 입력 카메라 수가 줄어드는 실험은 아니다.
카메라 수에 따른 자연스러운 개수 차이가 상한에 가려질 수 있으므로 원시 추론·변환 후·상한 적용 후의 개수를 구분한다.
상한이 없던 시험 결과는 현재 실행 계획의 결과와 합치지 않는다. 실행 시간은 실제 조건별 측정값으로 보고한다.

## 서버 준비

RTX A6000 48GB 등 NVIDIA GPU, Ubuntu 22.04 / CUDA 11.8 devel 환경을 사용한다.
원본 설치 파일은 `runpod/00_setup_pod.sh`, `01_install_gs.sh`, `02_install_da3.sh`, `03_get_data.sh`다.
새 터미널마다 다음을 실행한다.

```bash
cd /workspace/dgs_exp
source runpod/env.sh
```

처음 설치할 때는 순서대로 실행하고 각 단계의 성공을 확인한다.

```bash
bash runpod/00_setup_pod.sh
source runpod/env.sh
bash runpod/01_install_gs.sh
bash runpod/02_install_da3.sh
bash runpod/03_get_data.sh
```

오래 걸리는 작업은 tmux 안에서 실행한다. 브라우저를 닫아도 계속되지만 Pod를 정지하면 중단된다.
`/workspace` 볼륨과 결과는 서버 종료 방식에 따라 사라질 수 있으므로 내려받아 보관한다.

## 포함된 15개 실행 계획 사용

공식 Flame Steak 포즈, 시드 0, 기존 AABB, 300프레임 계획이 `jobs/camera15_cap1m_s0`에 들어 있다.

```bash
conda activate gs
bash jobs/camera15_cap1m_s0/preflight.sh

tmux new -s init15
bash jobs/camera15_cap1m_s0/run_initializations.sh
```

초기화 명령은 다음을 순서대로 수행한다.

1. GPU/패키지/코드 버전과 소스 revision lock 기록.
2. 3DGS 기준 초기화 학습.
3. 전체 학습 뷰의 DA3 초기화와 DA3+1k 생성.
4. 카메라 선택 12개 조건의 DA3 초기화와 1,000회 보정.
5. 보정 전후 PSNR·SSIM·LPIPS, 가우시안 수, AABB 내부 수, 공간 점유 지표 기록.

전체 뷰 DA3와 DA3+1k는 같은 DA3 원시 추론을 공유한다. 각각 별도 조건으로 평가한다.
평가 지표와 공간 진단은 학습 시간 측정 밖에서 계산한다.

단계마다 입력 설정·환경·출력 해시를 저장하므로, 같은 명령을 재실행하면 완료된 단계는 확인 후 건너뛴다.
부분 출력은 `.incomplete_...` 이름으로 보존한 후 재시도한다. 설정/코드/GPU가 달라지면 새 태그를 사용한다.
정렬 오차가 경험적 검토 기준을 넘으면 변환을 중단한다. 2% 기준은 품질 보장선이 아니며,
문제가 생기면 학습 시점의 기하/영상으로 검토한다. 진단용 override 결과는 정식 비교에서 제외된다.

첫 초기화가 준비된 뒤 대표 조건으로 후속 30프레임만 검증할 수 있다. 원본 실행 이름과 구분한다.

```bash
N_FRAMES=31 bash scripts/08_run_stream.sh smoke_largest /path/to/largest_init.ply
```

전체 초기화가 정상 완료됐고 메모리/속도를 확인한 다음 스트리밍을 시작한다.

```bash
bash jobs/camera15_cap1m_s0/run_streams.sh
tmux attach -t q
```

조건별 실패/완료 상태는 `jobs/camera15_cap1m_s0/queue_status.csv`에 기록한다.
큐는 실패한 조건을 표시하고 다음 조건으로 진행한다. `finished` 문구만으로 전체 성공으로 판단하지 않는다.
새 tmux 큐에도 AABB와 프레임 수를 명시적으로 전달한다.

## 평가와 CSV 저장

```bash
bash jobs/camera15_cap1m_s0/evaluate.sh
```

평가 전에 0~299 프레임 범위, 완료 로그, 최종 단계 float PSNR, 단계별 시간,
렌더링 파일과 NTC/추가 가우시안 파일을 검사한다.
미완료 조건이 있더라도 CSV는 생성하고 상태를 표시한다. 전체가 정상일 때 그림을 생성한다.

기본 결과 폴더는 `/workspace/runs/camera_capped/results`다.

| 파일 | 내용 |
|---|---|
| `summary.csv` | 15개 조건 전체의 핵심 결과와 상태. 미완료도 한 행으로 유지 |
| `summary_valid.csv` | 완료·지표·설정 일치 검사를 통과한 비교 대상. 시간 검증이 안 된 경우 시간 칸은 비움 |
| `initialization.csv` | 원시/변환 후/투입 개수, 보정 전후 품질, 초기화 단계별 시간 |
| `per_frame.csv` | 15×300=4,500행. 각 프레임의 품질·시간·가우시안 수·완료 여부 |
| `run_status.csv` | 완료 프레임 수, 누락/손상 프레임, 실패 단계와 검토 사유 |
| `experiment_conditions.csv` | 카메라 목록·시드·AABB 6개 좌표·조건·버전 |
| `experiment_conditions.json` | 상세 환경과 원본 메타데이터를 포함한 기록 |
| `spatial_diagnostics.csv` | 같은 복셀 안의 중심 수와 여러 뷰 출신 중심의 중첩 지표 |
| `profile_components.csv` | 선택적으로 실행한 짧은 NTC/렌더링/역전파/저장 프로파일 |
| `columns.csv` | 요약표 열의 뜻과 단위 |
| `collection.json` | 계획/완료/비교 가능 조건 수 |

CSV는 Excel에서 한글이 깨지지 않도록 UTF-8 BOM으로 저장한다.
빈칸은 미측정 또는 해당 없음이다. 실제 측정한 0과 구분한다.
`summary.csv`의 미검증 시간은 관측 기록으로만 남기고 `time_reportable=False`를 표시한다.

GPU 없이 수집만 다시 실행할 수도 있다. 아직 시작하지 않은 조건도 상태가 나오며 지표는 비어 있다.

```bash
python scripts/15_collect_results.py \
  --manifest jobs/camera15_cap1m_s0/experiment_manifest.json \
  --out-dir /workspace/runs/camera_capped/results
```

다운로드한 결과를 다른 컴퓨터에서 정리할 때는 원래 서버 경로를 매핑한다.
예를 들어 서버 `/workspace` 아래의 `runs` 전체를 `C:/my_experiment/runs`로 보관했다면:

```bash
python scripts/15_collect_results.py --manifest jobs/camera15_cap1m_s0/experiment_manifest.json \
  --path-map /workspace=C:/my_experiment --out-dir C:/my_experiment/results
```

## 측정값 해석

- `stream_psnr`: 최종 학습 단계 로그의 float PSNR 평균. 로그가 없으면 PNG PSNR로 대체하지 않는다.
- `stream_psnr_png`: 8-bit PNG에서 계산한 PSNR을 별도 보관한다.
- SSIM/LPIPS는 8-bit 렌더링으로 통일한다. 기본 LPIPS는 모든 프레임에 계산한다.
- `measured_pipeline_s`: 기록된 초기화+NTC warm-up+스트리밍 wall 합이다.
  카메라 보정·가중치 로딩·평가·공간 진단 시간을 제외하므로 실제 통화 지연이나 완전한 전체 실행 시간이 아니다.
- `time_reportable`: 필요한 시간값과 초기 GPU 단독 사용 점검이 모두 확인됐을 때만 True다.
  확인 불가능한 상태를 정상으로 바꾸지 않는다. 시작 시 점검이므로 실행 중 다른 작업도 함께 하지 않는다.
- GPU 메모리는 스트리밍 중 장치 전체 메모리를 2초 간격으로 관측한 값이다. 순간 피크를 놓칠 수 있다.
- 가우시안 개수는 생성/변환/실제 투입 단계를 구분한다. 상한으로 개수가 같아진 것을 뷰 수의 자연스러운 포화로 해석하지 않는다.
- 공간 중첩은 **중심 점유의 보조 지표**다. 가우시안의 실제 부피 교차나 불필요한 중복을 증명하지 않는다.
- 첫/마지막 30프레임 비교는 후속 프레임이 60개 이상일 때만 기록한다.
- cam00 품질은 결과 평가용이다. 평가 PSNR로 카메라 선택이나 정렬 설정을 튜닝하지 않는다.

## 짧은 구성 요소 프로파일 — 선택 사항

전체 15개 조건을 다시 돌리지 않는다. 기존 조건을 몇 프레임만 별도 출력 폴더로 재생한다.

```bash
conda activate gs
python scripts/21_profile_stream.py \
  --run-dir /workspace/runs/stream/camera15_cap1m_s0_position_fps_18_da3_1000 --frames 3
```

`run/diagnostics/profile_events.csv`에 NTC forward, render, backward, PLY/NTC 저장 시간을 기록한다.
현재·직전 연산의 GPU 작업이 섞이지 않도록 동기화하므로 실제 처리 속도가 바뀐다.
이 값은 원인 분석 전용이고 정식 처리 시간과 합치거나 대체하지 않는다.
프로파일 이후 `15_collect_results.py`를 다시 실행하면 통합 CSV에 포함된다.
CUDA 부분은 이 컴퓨터에서 실행하지 않았으므로 서버에서 검증해야 한다.

## 버전 재현

설치 후 초기화 시작 시 `jobs/camera15_cap1m_s0/environment_gs.json`, `environment_da3.json`, `versions.lock.json`을 기록한다.
정상 실행을 확인한 뒤 이 파일들을 결과와 함께 보관한다.
다음 서버 설치에서는 `VERSIONS_LOCK`에 보관한 lock 경로를 지정하면 3DGStream·DA3·tiny-cuda-nn 소스 commit을 고정한다.
패키지 목록은 환경 JSON의 pip_freeze에도 기록한다. 첫 설치의 최신 소스가 GPU 환경과 호환되는지는 짧은 실행으로 검증한다.

```bash
export VERSIONS_LOCK=/workspace/saved/versions.lock.json
bash runpod/01_install_gs.sh
bash runpod/02_install_da3.sh
```

## 새 계획 생성

기본 AABB와 카메라 목록을 바꾸지 않을 때는 포함된 실행 계획을 사용한다.
개수 제한 등 설정을 바꿀 때는 새 태그와 새 출력 폴더로 생성한다.

```bash
python scripts/14_make_camera_jobs.py \
  --plan configs/flame_steak_camera_plan/selection_plan.json \
  --aabb configs/aabb_original.json --tag camera15_capped \
  --server-root /workspace/runs/camera_capped --max-gaussians 1000000 \
  --out-dir /workspace/runs/camera_jobs_capped
```

일반적으로 15개를 생성한다. 기존 기준 조건을 별도로 다룰 필요가 있을 때만 `--only-camera-conditions`로 12개를 생성한다.
AABB를 여러 개로 복제해 실험 횟수를 늘리는 동작은 없다.

## 2026-11-22 수정: 정렬 실패 조건 건너뛰기 (v4)

- 무작위 4뷰 조건에서 DA3 정렬 잔차가 11.83%로 기준 2%를 넘어 전체 실행이 멈춘 문제를 고쳤다.
- `jobs/camera15_cap1m_s0/run_initializations.sh`는 이제 조건마다 따로 실행된다. 한 조건이 실패하면
  `alignment_failed`(정렬 기준 초과) 또는 `failed`로 기록하고 다음 조건으로 넘어간다.
  실패 목록은 `jobs/camera15_cap1m_s0/initialization_failures.tsv`에 남는다.
  변환 도구: `python scripts/22_isolate_initializations.py <run_initializations.sh>`.
- `scripts/run_queue.sh`는 첫 프레임이 없는 조건의 스트리밍을 건너뛰고 원래 실패 사유를 유지한다.
- `summary.csv`·`run_status.csv`의 `status`에 `alignment_failed`/`skipped_no_init`가 표시되고,
  정렬 실패 조건도 `alignment_residual_rel`(정렬 잔차)이 채워진다.
- 그림(`10_report.py --skip-unfinished`)은 끝난 조건만으로 그린다.
- 건너뛴 조건이 있으면 드라이버는 마지막에 exit 1(검토 필요)로 끝나고 `pipeline.stage`에 `failed: evaluation`이
  찍힐 수 있다. 이 경우에도 끝난 조건의 CSV와 그림은 정상이며, 서버는 10분 뒤 자동 Stop된다.
- 이미 끝난 단계는 완료 표시(`*.done.json`)가 있어 다시 실행하지 않는다.
