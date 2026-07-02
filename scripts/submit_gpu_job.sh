#!/usr/bin/env bash
#SBATCH --partition=shared_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=crelat

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch scripts/submit_gpu_job.sh <python-script> [args...]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec ./scripts/run_gpu_container.sh python "$@"
