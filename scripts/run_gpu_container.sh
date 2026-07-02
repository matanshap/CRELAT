#!/bin/bash
# Run a Python command inside the CUDA Apptainer image with the persistent conda env.

set -euo pipefail

IMAGE=/truenas/sif_images/pytorch_cuda12.6_ngc_conda_vscode.sif
ENV_PATH=/conda_envs/crelat_gpu
PROJECT_ROOT=/truenas/home/shapirma/CRELAT

if [ "$#" -eq 0 ]; then
  set -- python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
fi

# The persistent conda environment supplies the container runtime, while the
# project virtual environment supplies CRELAT's verified Python dependencies.
if [ "${1:-}" = "python" ] && [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  shift
  set -- /workspace/.venv/bin/python "$@"
fi

/usr/bin/apptainer exec --nv \
  --bind /truenas/home/shapirma/conda_envs:/conda_envs \
  --bind "$PROJECT_ROOT":/workspace \
  "$IMAGE" \
  bash -lc "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate $ENV_PATH
    cd /workspace
    export PYTHONPATH=/workspace/src:\${PYTHONPATH:-}
    export MPLCONFIGDIR=/tmp/crelat-matplotlib
    exec \"\$@\"
  " bash "$@"
