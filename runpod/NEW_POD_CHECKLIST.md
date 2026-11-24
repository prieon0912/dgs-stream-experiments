# 새 pod 준비 체크리스트 (가지치기 실험)

## 1. pod 설정 (RunPod Deploy 화면)
- GPU: RTX A6000 × 1 (On-Demand, 약 $0.69/hr)
- 컨테이너 이미지: **`runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`**
  - 템플릿 제목이 "CUDA 13.2"로 보여도 Workload 줄의 이미지가 위 값이면 된다(Set overrides).
  - 3DGStream 환경(torch 2.0.1 + cu118 확장 빌드)에 nvcc 11.8이 필요하다.
- Container disk 50GB, Persistent storage는 **Volume disk 200GB** (`/workspace`)
  - 크기 입력칸은 지우고 다시 쓰면 자릿수가 잘린다. 끝에 숫자를 덧붙여 200으로 맞춘 뒤 값을 확인한다.
- 합계 약 $0.72/hr, 정지 시 $0.056/hr.

## 2. 코드 올리기 (확인된 방법)
1. 노트북의 `deliverables/dgs_exp_v5_prune_update.zip`을 세션으로 가져온다 (device_stage_files).
   - 가져온 파일(/mnt/user-data/uploads/...)만 브라우저 업로드가 허용된다. outputs 폴더나 Windows 경로를 바로 쓰면 거부된다. (2026-11-23 확인)
2. 새 pod의 JupyterLab 탭(콘솔의 "Jupyter Lab" 링크로 열어야 로그인됨)에서 임시 `<input type=file>`을 만들고 file_upload로 zip을 넣는다.
3. 그 파일을 Jupyter contents API(`PUT /api/contents/workspace/<zip>`, base64)로 저장한다.
4. 터미널에서:
   ```bash
   python3 -m zipfile -e /workspace/dgs_exp_v5_prune_update.zip /workspace/ && bash /workspace/dgs_exp_v5/runpod/bootstrap_new_pod.sh
   ```
   - nvcc 11.8을 확인하고 tmux 세션 `prune`에서 `run_prune_update.sh`를 시작한다.
   - 진행: `cat /workspace/prune_update.stage`, `tail -f /workspace/prune_update.log`

예비 방법: 사용자가 JupyterLab 파일 목록(`/workspace`)에 zip을 끌어다 놓는다.

## 3. 기존 pod
- `flame-steak-camera15-migration-2`(wb3wy5xrxrb45m)는 건드리지 않는다.
