---
trigger: always_on
---

This repo is on the BGU ECE HPC cluster at `/truenas/home/shapirma/CRELAT`.

For normal lightweight CPU work, prefer the existing local Python setup if it is enough for the task.

For any task that needs CUDA, PyTorch GPU, embeddings, transformers, or substantial model inference, choose the GPU path:

```bash
srun -p shared_a6000 --gres=gpu:1 --cpus-per-task=4 --mem=8G --time=01:00:00 ./scripts/run_gpu_container.sh python your_script.py
```

The GPU helper runs inside the Apptainer CUDA image:

```text
/truenas/sif_images/pytorch_cuda12.6_ngc_conda_vscode.sif
```

It activates the persistent conda environment:

```text
/truenas/home/shapirma/conda_envs/crelat_gpu
```

Verified CUDA setup:

```text
torch 2.9.0+cu128
cuda_available True
device NVIDIA RTX A6000
```

Do not run GPU work directly on the login node. Use `srun` or submit `scripts/submit_gpu_job.sh` with `sbatch`.
