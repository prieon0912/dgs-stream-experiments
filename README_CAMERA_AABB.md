# 카메라 선택과 AABB

전체 실행은 [README_RESULTS.md](README_RESULTS.md)를 따른다.

## 선택법

공식 Flame Steak의 COLMAP 외부 파라미터에서 카메라 위치와 시선을 읽었다.
21개 중 cam00을 평가용으로 제외한 20개에서 선택한다.

- `random`: cam01 공통 기준을 포함하고 나머지를 고정 시드로 무작위 추출.
- `position_fps`: 선택된 카메라들과의 최소 위치 거리가 가장 큰 후보를 차례로 추가.
- `direction_fps`: 선택된 시선들과의 최소 각도가 가장 큰 후보를 차례로 추가.

모든 방법에서 cam01을 DA3의 첫 입력으로 사용한다.
동일 방법 안에서는 4뷰가 8뷰에, 8뷰가 12뷰에, 12뷰가 18뷰에 포함된다.
위치/각도 기준은 기하학적 분산이며, 실제 가림이나 영상 품질의 최적성을 보장하지 않는다.
시선은 COLMAP/OpenCV 카메라 +Z를 월드 좌표로 변환한 벡터다.

`configs/flame_steak_camera_plan/selection_summary.csv`에 실제 선택 목록이 있다.
이번 시드 0의 12개 조합은 모두 서로 다르다. 다른 설정에서 중복되면 `same_view_set_as`로 표시한다.
원본 출처/해시는 같은 폴더의 `SOURCE.json`에 기록했다.

```bash
python scripts/12_select_cameras.py \
  --cameras-json configs/flame_steak_camera_plan/cameras.json \
  --counts 4 8 12 18 --seed 0 --out-dir /workspace/runs/new_camera_plan
```

직접 COLMAP을 읽으려면 `--cameras-json` 대신 `--frame-dir`을 사용한다.
필요한 파일은 `sparse/0/images.bin`이며 GPU나 영상 전체는 필요 없다.
실제 실험 데이터의 포즈가 공식 suite와 다르면 그 포즈로 목록을 다시 생성한다.

## 공통 AABB

초기 경계를 자동 추정하거나, AABB 여러 개를 교차 비교하지 않는다.
현재 최소 `(-20,-15,5)`, 최대 `(15,10,23)`을 15개 조건에 공통 적용한다.
좌표 단위는 COLMAP 월드 좌표이며 반드시 미터인 것은 아니다.

```bash
python scripts/13_aabb.py manual --min -20 -15 5 --max 15 10 23 \
  --out /workspace/runs/aabb_shared.json

python scripts/13_aabb.py stats --aabb /workspace/runs/aabb_shared.json \
  --points /path/to/baseline.ply /path/to/da3.ply --out /workspace/runs/aabb_counts.json

python scripts/13_aabb.py inspect-ntc --ntc /path/to/ntc.pth
```

`AABB_CONFIG`를 정적 평가와 NTC warm-up에서 공유하고, 저장된 NTC의 실제 경계를 다시 확인한다.
설정·입력·코드/GPU 환경이 달라진 상태로 같은 실행 폴더를 재개하는 것을 막는다.
기존 3DGStream처럼 상자 밖 가우시안은 렌더링에는 남고 NTC 움직임 갱신에서는 제외된다.
내부 비율은 실제로 움직인 가우시안 비율이 아니다.

본 파이프라인의 카메라 보정·초기 공간 정보에 따른 AABB 결정 기능은 이번 비교에 포함하지 않는다.
