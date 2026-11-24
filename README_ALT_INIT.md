# 세 번째 실험: DA3 대신 다른 첫 프레임 생성 방법 (dgs_exp_v6)

v5(가지치기 실험)를 복사한 폴더다. v5 파일은 바꾸지 않았고 새 파일만 추가했다.
두 번째 실험 서버(`8cdp3j2szanori`)의 볼륨에 있는 것(gs 환경, 데이터, 가지치기 결과)을 그대로 이어 쓴다.

## 비교 대상 (0번 프레임만)

| 조건 | 무엇을 하나 | 개수 | 보정 |
|---|---|---|---|
| 기준: DA3 + opacity 1M | 두 번째 실험의 기준 조건(같은 서버의 결과를 읽기만 함) | 100만 | 1k |
| `da3ref` (기본 꺼짐, `DA3REF=1`로 켬, 약 2분) | 두 번째 실험의 DA3 opacity 100만 선택 결과를 이 기계에서 다시 1k 보정. 저장된 보정 시간과 비교해 서버 이전 전후 기계 속도 차이를 잰다 | 100만 | 1k |
| `3dgs15k` (기본 꺼짐) | 우리 3DGS 15k 기준선을 이 서버에서 다시 실행(`WITH_BASELINE=1`). 이번에는 쓰지 않고 기계 차이는 `da3ref`로 확인 | 자동 | 없음(학습 자체) |
| `taming` | Taming 3DGS: 3DGS 15k와 같은 일정, 개수 늘리기 규칙만 Taming(`--mode final_count --budget N --sh_lower`). 고불투명도 전환(`--ho_iteration`)은 끔: 켜면 PLY에 sigmoid가 아닌 abs 불투명도가 저장돼 3DGStream이 잘못 읽음 | 거의 정확히 N | 없음(학습 자체) |
| `edgs` | EDGS: RoMa 대응점 → 삼각측량 초기화(공식 코드 경로) → 우리 1k 보정 | 정확히 N | 1k (DA3와 동일) |
| `zipsplat` | ZipSplat(504px 가중치) + TokenBA → 예측 카메라 중심으로 COLMAP 좌표에 Sim3 정렬 → 우리 1k 보정 | `--compressions`로 조절(정확한 N 불가) | 1k (DA3와 동일) |

기본 N = 100만/80만/60만/40만(가지치기와 같음), ZipSplat 압축비 = 1.0/0.5.

## 공정성 규칙
- **같은 20개 학습뷰(cam01~cam20)**: cam00을 뺀 학습용 장면 복사본(`scripts/30_make_train_scene.py`)을 Taming과 EDGS에 넣고,
  각자의 평가 분할은 끈다(Taming `--eval` 미사용, EDGS `do_train_test_split=False`).
  원래대로 두면 두 코드 모두 "8개마다 1개"를 빼서 cam00이 아닌 뷰를 버린다.
- **같은 해상도와 SH**: `images_2`, SH 차수 1(3DGStream 기준).
- **같은 평가**: `04_check_and_eval.py --train`(cam00 + 학습뷰 평균), `17_record_initialization.py`, 같은 AABB.
- **시간**: 가중치 로딩, COLMAP 제외(기존 규칙). Taming/3DGS는 train.py 벽시계(시작 시간 포함, 기존 기준선과 같은 규칙).
  EDGS는 `init_with_corr` 시간 + 1k 보정, ZipSplat은 추론+TokenBA+정렬 시간 + 1k 보정.
- **판정**: 기준(DA3 opacity 1M + 1k) 대비 cam00과 학습뷰 PSNR이 **둘 다** t 이내면 통과, t = 0.1/0.3/0.5 dB.
  한 번씩만 실행(seed 0), cam00은 검증 뷰.

## 환경 (서로 섞이지 않게 env 분리)
| env | torch | 이유 |
|---|---|---|
| `taming` | 2.0.1+cu118 | gs 환경·3DGS 기준선과 같은 torch → 시간 비교 공정 |
| `edgs` | 2.4.1+cu118 | RoMa + 최신 graphdeco 코드 |
| `zipsplat` | 2.4.1+cu118 + xformers 0.0.28.post1 | DA3-Giant 백본(da3 환경과 같은 조합) |

Taming과 EDGS는 각각 `diff_gaussian_rasterization`, `simple_knn`이라는 **같은 이름**의 CUDA 확장을 따로 빌드한다.
3DGStream 것과 이름이 같으므로 절대 gs 환경에 설치하지 않는다(설치 스크립트는 gs/da3 환경을 건드리지 않음).
출력은 모두 표준 3DGS PLY라서 보정·평가는 gs 환경에서 같은 코드로 한다.

커밋 고정: `alt_init/alt_env.sh` (Taming fd0f7d9, EDGS f90b022 → RoMa 3701174 / gaussian-splatting 54c035f, ZipSplat 73ab63f).
EDGS의 `.gitmodules`는 `git@github.com` 주소라 https로 바꿔서 받는다. simple-knn(gitlab.inria.fr)이 안 되면 gs와 같은 github 미러를 쓴다.

## 실행
```bash
# 두 번째 서버에서 (tmux 안)
cd /workspace && unzip -o dgs_exp_v6_alt_init.zip   # → /workspace/dgs_exp_v6
bash /workspace/dgs_exp_v6/runpod/run_alt_init.sh   # 설치 → 스크린 전체
#   STOP_POD=1 bash ...                             # 끝나면 stop(terminate 아님)
#   SCREEN_ARGS="--methods edgs --n 600000" bash ... # 일부만
cat /workspace/alt_init.stage
```
결과: `/workspace/runs/alt_init/results/alt_init.csv` (조건별 개수, cam00/학습뷰 PSNR과 기준 대비 차이, 통과 여부, AABB 내부 수, 시간, 정렬 오차).

## 알려진 위험
- **ZipSplat**: 개수를 정확히 정할 수 없다. 논문 수치(252px·16뷰·압축비 1.0에서 16.6만 개, 토큰당 약 32개)로 어림하면 504px·20뷰에서 압축비 1.0은 수십만 개(약 60만 개 안팎) 정도일 것으로 보지만 실제 개수는 실행해서 기록한다. 학습 해상도가 우리보다 낮고 우리 1k 보정은 개수를 늘리지 않으므로 34 dB에 못 미칠 수 있다.
  카메라를 직접 넣는 옵션은 현재 코드의 고수준 API에 없어서, 예측 카메라 중심을 COLMAP 중심에 맞추는 Sim3로 좌표를 맞춘다
  (중앙 잔차/카메라 배치 반경 > 5% 또는 회전 오차 중앙값 > 5°이면 `_REVIEW.ply`로 옮기고 사용하지 않음). 가중치는 CC BY-NC 4.0(비상업).
- **ZipSplat 안전장치**: 실행 전에 예상 개수(입력 크기/패치 14 × 뷰 수 × 압축비 × 토큰당 32개, 20뷰·압축비 1.0이면 약 62만 개)를 출력한다. 실제 개수가 상한(`--max_gaussians`, 기본 100만)을 넘으면 DA3 상한과 같은 규칙(불투명도 상위)으로 자르고 잘린 수를 기록한다. 입력 뷰가 학습 범위(2~24) 밖이면 실행하지 않는다. NaN/inf 가우시안은 빼고 기록하며 1%를 넘으면 실패 처리한다. GPU 메모리 부족은 설정을 바꿔 자동 재시도하지 않고(방법이 달라지므로) `_zipsplat_failed.json`에 이유를 남기고 멈춘다.
- **EDGS**: 초기 점에 잡음이 있다. 목표의 1.5배를 만들고, EDGS가 투영 오차로 보이지 않게 만든 점(opacity logit −10)은 버린 뒤 나머지에서 seed 0 무작위로 정확히 N개를 고른다. 보이는 점이 N개보다 적으면 그 N은 **제외**하고(채워 넣지 않음, `excluded.json`), 보이는 점을 **전부 쓴 버전**을 `edgs_all` 조건으로 넣는다(제외된 실행이 여러 개면 보이는 점이 가장 많은 것).
- **Taming**: `final_count` 일정은 15k 끝 무렵 목표에 도달하도록 짜여 있어 최종 개수가 N보다 약간 적을 수 있다(실제 개수를 기록).
- 설치는 서버에서 처음 해 보는 것이라 빌드 오류가 날 수 있다. 각 단계는 다시 실행하면 끝난 부분을 건너뛴다.

## 이번 실행의 결정 (2026-11-24)
- 이 실험은 후보를 추리기 위한 **사전 실험**이다. 서버 이전으로 GPU 기계가 바뀌어도 시간 보정은 하지 않고 그대로 이어 붙인다
  (`da3ref`, `3dgs15k` 모두 끔). 보고서에는 "시간은 같은 GPU 모델(A6000)의 다른 기계에서 측정"이라고만 적는다.
- **실패 처리**: 한 방법이 설치나 실행에서 실패해도 나머지는 계속 돈다. 설치는 방법별로 따로 하고(`install_status.txt`, `install_<방법>.log`),
  설치된 방법만 스크린에 넣는다. 조건 하나가 실패해도 다음 조건으로 넘어간다. 모든 방법에 영향을 주는 문제(GPU 없음, 데이터·디스크 부족)만 전체를 멈춘다.
  `STOP_POD=1`이면 성공이든 실패든 끝날 때 서버를 stop한다.
