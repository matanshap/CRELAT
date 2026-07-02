# HPC Execution

Do not run CUDA, embedding generation, Transformers inference, or substantial model inference on the login node.

The container helper activates the persistent GPU conda environment and routes `python` commands through the repository `.venv`, whose dependencies were validated in Slurm.

Interactive Slurm execution:

```bash
srun -p shared_a6000 --gres=gpu:1 --cpus-per-task=4 --mem=8G \
  --time=01:00:00 ./scripts/run_gpu_container.sh python \
  pipelines/build_interactions.py \
  --config configs/experiments/genre-analysis.yaml
```

Batch execution:

```bash
sbatch scripts/submit_gpu_job.sh pipelines/build_interactions.py \
  --config configs/experiments/genre-analysis.yaml
```

Check jobs before and after execution with `squeue -u shapirma`. Transformer adapters reject inference outside Slurm unless `CRELAT_ALLOW_LOGIN_MODEL=1` is explicitly set for a controlled non-cluster environment.

Semantic indexing for researcher corpora also requires Slurm:

```bash
srun -p shared_a6000 --gres=gpu:1 --cpus-per-task=4 --mem=8G \
  --time=01:00:00 ./scripts/run_gpu_container.sh python \
  research/skills/research-question-interlocutor/scripts/build_index.py \
  research/researchers/andrew-piper --semantic
```
