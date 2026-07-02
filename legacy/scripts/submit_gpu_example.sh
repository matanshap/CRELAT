#!/bin/bash
#SBATCH --job-name=crelat_gpu
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=shared_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail

IMAGE=/truenas/sif_images/pytorch_cuda12.6_ngc_conda_vscode.sif
ENV_PATH=/conda_envs/crelat_gpu
PROJECT_ROOT=/truenas/home/shapirma/CRELAT

/usr/bin/apptainer exec --nv \
  --bind /truenas/home/shapirma/conda_envs:/conda_envs \
  --bind "$PROJECT_ROOT":/workspace \
  "$IMAGE" \
  bash -lc "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate $ENV_PATH
    cd /workspace
    python - <<'PY'
import torch

print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0))
PY
  "
