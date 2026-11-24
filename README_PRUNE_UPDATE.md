# 가지치기 4종 × AABB 내부 갱신 3종 실험 (dgs_exp_v5)

기존 `dgs_exp`(v4, 카메라 15조건)를 복사한 폴더다. 기존 파일은 거의 그대로 두고 새 파일만 추가했다
(유일한 수정: `16_snapshot_environment.py` 코드 해시에 `stream/` 포함). 기존 서버·폴더와 섞지 말 것.

## 실험 구성

| 단계 | 내용 | 결과 |
|---|---|---|
| 1단계 (0번 프레임만) | 가지치기 4종 × **직접 정한 개수 N**(기본 100만): 선택 → 1k 보정 → cam00·학습뷰 품질, AABB 내부 수. 결과를 보고 N을 늘리거나 줄여 추가로 돌리면 표에 누적된다 | `results/phase1.csv`, 방법별 경향과 허용 N 제안 |
| 파일럿 (2단계 직전) | 1단계에서 **고른 N**의 초기화(기본: 불투명도 방법의 N_opacity)로 모드 A 30프레임 스트리밍, 가우시안별 이동량 저장 | `B_TAU_XYZ`를 phase2.env에 자동 추가 |
| 2단계 | 가지치기 4종(각자 고른 N) × 갱신 모드 A/B/C, 300프레임 = 12회 | `results/phase2.csv` |

가지치기 (`scripts/23_select_gaussians.py`, 핵심 함수는 `common/selection.py`, `common/redundancy.py`)
- `opacity` 불투명도 상위 N (지금까지 쓰던 방식, 1M이 기준 조건)
- `voxel` 복셀 격자에서 칸마다 불투명도 최대 1개만 남김. 남는 칸 수가 N 이상이 되는 가장 큰 복셀 크기를 이분 탐색으로 찾고, 넘치는 만큼은 불투명도로 잘라 정확히 N개
- `contrib_rank` 렌더 기여도 상위 N (LightGaussian 방식)
- `contrib_sample` 기여도에 비례한 확률로 N개 비복원 추출 (Mini-Splatting 방식, seed 0)
- `redundancy` (추가) Papantonakis et al. 2024의 공간 중복도 점수를 공식 코드에서 그대로 옮겨 계산하고(학습 카메라 기준 픽셀 크기 구, 30-NN, 확대 타원 포함 판정, 포함한 구들 중 최솟값, 임계값 max(평균+1·표준편차, 3)), 후보 중 불투명도 하위(공식: 후보 중앙값 미만)를 제거하는 과정을 목표 개수까지 반복한다. 후보가 없으면 남은 수는 불투명도로 제거하고 그 수를 기록한다. 논문 점수를 활용한 변형이다.
- `redundancy_once` (추가) 같은 공식 중복도 점수를 원본 전체에 한 번만 계산하고, 후보(임계값 초과)를 불투명도 낮은 순으로 먼저 제거한 뒤 나머지를 불투명도로 제거한다. 반복 재계산이 없어 빠르지만 제거 후 점수가 갱신되지 않는다. (`redundancy`도 제거 순서가 N과 무관하므로 한 번 계산해 캐시하고, N마다 잘라 쓴다. 결과는 반복 방식과 동일하며 테스트로 확인했다. 기록되는 선택 시간은 처음 계산한 시간이다.)
- `reproj_dedup` (추가) DA3 뷰 순서대로, 앞선 뷰에서 남긴 가우시안을 현재 뷰에 z-buffer로 투영해 같은 픽셀(3×3 셀)의 가장 앞 표면과 상대 깊이 2% 이내로 일치하는 현재 뷰 가우시안을 중복으로 본다. 앞에 있는 물체·뒤 표면은 중복으로 보지 않는다. 대표 하나만 남기고 속성은 합치지 않는다. 그 뒤 불투명도로 정확히 N개를 맞추고, 순수 중복 수와 추가로 넣거나 뺀 수를 기록한다. FreeSplat의 투영·깊이 대응 아이디어를 DA3 출력에 적용한 별도 변형이다.

기여도 = 학습뷰 20개 전체 픽셀에서 α·T의 합. 기존 래스터라이저로 colors_precomp=1로 렌더하고
이미지 합을 색에 대해 미분하면 그 값이 정확히 Σα·T라서 별도 CUDA 빌드가 필요 없다.
LightGaussian 기본값대로 부피 항 (V/V₉₀%)^0.1을 곱한다(`--volume_power 0`이면 끔).
점수는 원본 PLY에만 의존하므로 한 번 계산해서 `scores/contrib_scores.npz`에 캐시하고,
선택 시간에는 처음 계산한 시간을 그대로 더해 기록한다.

갱신 모드 (`stream/train_frames_select.py`, 논리는 `stream/selective_update.py`)
- A: AABB 안 전부 갱신 = 원래 3DGStream. 로그만 추가되고 계산은 같다.
- B: 이동 이력 고정 (CPOStream 방식). 1–10프레임과 30의 배수 프레임은 전부 갱신하고, 그 외에는 최근 10프레임 동안 실제로 갱신됐던 프레임의 |Δxyz| 최댓값이 `B_TAU_XYZ`를 넘는 가우시안만 갱신한다. 회전 임계값 `B_TAU_ROT_DEG`는 기본적으로 끔.
- C: 프레임 차이 마스크 (DecoGS 방식). 학습뷰마다 |I_t − I_{t−1}|의 RGB 최댓값 > 25/255인 픽셀을 반경 48로 max-pooling 팽창한다(DecoGS와 같은 정사각형 커널). 가우시안 중심을 투영해서 마스크 위에 오는 뷰가 2개 이상이면 갱신한다.
- 공통: 선택된 수가 128개 미만이면 그 프레임은 전체 갱신으로 대체하고 `fallback`으로 기록한다. 2단계(새 가우시안 추가)는 세 모드 모두 동일하다.

3DGStream 원본 저장소는 수정하지 않는다. 실행할 때 `Scene.__init__`, `training_one_frame_setup`,
`NeuralTransformationCache.forward`, `update_by_ntc` 네 군데만 갈아끼운다.
속도 비교의 주 지표는 프레임당 학습 시간(1단계+2단계, `train_sec_per_frame`)이다. 불러오기 시간이 빠져서 저장소 상태의 영향을 받지 않는다. A/B/C의 선택 시간은 1단계 전에 돌므로 `select_sec_mean`으로 따로 기록하고, 모드 비교 때는 `train_plus_select_sec_per_frame`을 본다. 선택 시간은 upstream "Frame … finished in" 시간 안에도 들어간다(t_prep). 모드 B의 재개용 상태 저장
시간(`io_sec`)은 따로 기록해서 `algo_sec_per_frame`에서 뺀다.

## 새 서버에서 실행

1. A6000 *devel* 템플릿으로 새 pod를 만들고, 이 폴더를 `/workspace/dgs_exp_v5`에 올린다.
2. `tmux new -s exp` 후 다음을 실행한다.
   ```bash
   cd /workspace/dgs_exp_v5
   bash runpod/run_prune_update.sh                   # 설치·데이터·DA3 원본·1단계(100만·80만·60만·40만 × 4방법)까지 한 뒤 멈춤
   # 경향을 보고 다른 개수를 추가로 시험 (결과는 phase1.csv에 누적):
   bash jobs/prune_update_s0/10_phase1_screen.sh --n 600000
   bash jobs/prune_update_s0/10_phase1_screen.sh --n 800000,700000 --methods voxel,contrib_rank
   # 방법별 N을 정해서 jobs/prune_update_s0/phase2.env 작성 (N_opacity= … N_contrib_sample=; phase2.env.suggested 참고)
   bash runpod/run_prune_update.sh                   # 파일럿(고른 N, B_TAU_XYZ 자동 추가) → 12회 스트리밍 → 평가
   ```
   검토 없이 제안된 N으로 바로 이어서 돌리려면 `AUTO_PHASE2=1`, 다 끝난 뒤 pod를 정지하려면 `STOP_POD=1`을 붙인다.
   정지(stop)만 하고 삭제(terminate)는 하지 않는다.
3. 진행 상태는 `/workspace/prune_update.stage`에 있다. 각 run의 `selection.jsonl`에는 프레임마다 NTC 입력 수가 기록된다.

원본 PLY(DA3 전체 뷰, 상한 적용 전): 새 pod에서 `00_prepare_source.sh`가 카메라 실험과 같은 설정으로
다시 만든다. 기존 서버 것을 쓰고 싶으면 `camera15_cap1m_s0_shared_all_views/`의
`converted.ply`, `converted_timing.json`, `converted_sources.npz`, `converted_alignment.json`을
`/workspace/runs/prune_update/source_all_views/`에 넣으면 이 단계를 건너뛴다.
어느 경우든 1단계에서 불투명도 100만 조건을 다시 측정해서 기준으로 쓴다.

## 판단 기준 (바꿀 수 있음)
- 허용 N: 1k 보정 후 **cam00 PSNR과 학습뷰 20개 평균 PSNR이 둘 다** 불투명도 100만 대비 0.3 dB 이내인, 시험해 본 N 중 가장 작은 값. 표에는 0.1 / 0.3 / 0.5 dB 각각의 통과 여부(o/x)를 같이 낸다. 시험한 N만 고려하므로, 출력의 안내("try a smaller N" / "boundary is between them" / "try a larger N")를 보고 다음 N을 정한다.
  - 근거: LightGaussian(66% 가지치기 + 회복 후 PSNR 유지, 70%부터 급락), Mini-Splatting(6.1M→0.6M 직접 가지치기 −0.3 dB), GaussianSpa(85% 제거 시 −0.2~−0.5 dB). 0.1~0.3 dB는 품질 유지, 0.5 dB는 눈에 띄는 손실로 보는 편이다.
- `B_TAU_XYZ`: 파일럿 이동량으로 모드 B를 오프라인 재현했을 때, 고정 프레임에서 AABB 내부의 약 35%가 갱신되는 값 (DecoGS가 약 35%만 갱신; `26_suggest_thresholds.py --target-active`, 0.5까지가 보수적 범위). DASS가 flame steak에서 측정한 프레임당 이동량이 대부분 0.01 이하였으므로, 제안값이 0.01보다 훨씬 크면 점검한다. 정한 값은 4가지 가지치기에 똑같이 적용한다.
- C: DecoGS 기본값 τ=25, r=48, κ=2 (같은 1352×1014 해상도, 같은 중심 투영 방식) 그대로.

## 파일
- `common/selection.py`, `scripts/23_select_gaussians.py`: 가지치기
- `stream/selective_update.py`, `stream/train_frames_select.py`, `scripts/08b_run_stream_select.sh`: 갱신 모드 A/B/C
- `scripts/24_summarize_phase1.py`, `25_summarize_phase2.py`, `26_suggest_thresholds.py`: 표와 제안값
- `jobs/prune_update_s0/`: `config.sh`, `00_prepare_source.sh`, `10_phase1_screen.sh`, `20_pilot_motion.sh`, `30_phase2_streams.sh`, `40_evaluate.sh`
- `runpod/run_prune_update.sh`: 전체 드라이버
- `tests/test_selection.py`, `tests/test_selective_update.py`: CPU 테스트. 가짜 3DGStream 모듈로 훅을 끝까지 돌려 본다.

GPU에서만 확인할 수 있는 부분(기여도 계산, 실제 3DGStream 훅)은 아직 실행해 보지 않았다.
새 pod에서 처음 돌릴 때 1단계 첫 `contrib_rank` 로그와 파일럿 첫 프레임의 `selection.jsonl`을 먼저 확인한다.
