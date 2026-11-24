> **dgs_exp_v5**: 가지치기 × AABB 내부 갱신 실험은 [README_PRUNE_UPDATE.md](README_PRUNE_UPDATE.md)를 본다. 아래는 v4 카메라 실험 설명이다(이 폴더에는 `jobs/camera15_cap1m_s0`만 남겼다).

# Flame Steak: 15개 조건 비교 및 CSV 결과 정리

실행과 CSV 설명은 [README_RESULTS.md](README_RESULTS.md)를 기준으로 진행한다.

- 기존 기준 조건 3개: 직접 학습한 3DGS / 전체 학습 뷰 DA3 / 전체 학습 뷰 DA3+1k.
- 카메라 선택 12개: 4·8·12·18뷰 × 무작위·위치 거리·시선 각도.
- 수동으로 정한 같은 AABB 하나를 15개 조건에 적용한다.
- 현재 실행 계획은 모든 DA3 초기화에 최대 1,000,000개 상한을 적용한다. 불투명도가 높은 순서로 남기며, 원래 더 적으면 늘리지 않는다.
- 보정 전후 품질, 가우시안 수, AABB 내부 비율, 처리 시간, 완료 여부를 CSV로 정리한다.
- 미완료·측정 누락·다른 설정/GPU가 섞인 결과는 비교 대상에서 구분한다.

현재 실행 계획은 `jobs/camera15_cap1m_s0`이다. 공식 카메라 좌표·시드 0·기존 AABB·300프레임을 사용한다.
`jobs/camera15_s0`는 상한이 없던 이전 계획을 기록용으로 보존한 것이므로 현재 실험과 혼용하지 않는다.
`examples/results_before_run`에는 실행 전 CSV 양식이 있다. 실제 측정값은 비어 있다.

기존 카메라 선택/AABB 기능은 [README_CAMERA_AABB.md](README_CAMERA_AABB.md)에 설명했다.
과거 버전의 실행 안내는 [README_LEGACY.md](README_LEGACY.md)에 보존했다.
현재 조건 생성에는 `--max-gaussians 1000000`을 명시한다. 상한 적용 전후의 개수는 별도로 기록된다.

RunPod RTX A6000 / CUDA 11.8에서 DA3 추론·100만 개 제한·1,000회 보정·짧은 스트리밍을 검증했다.
이는 전체 15개 조건의 완료를 의미하지 않는다. 전체 진행 상태는 서버의 `/workspace/pipeline.stage`와 `/workspace/pipeline.log`에서 확인한다.
