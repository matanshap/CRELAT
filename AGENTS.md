# Agent Notes For CRELAT

This repo lives on the BGU ECE HPC cluster at:

```text
/truenas/home/shapirma/CRELAT
```

Before running anything nontrivial, read the project agent rules:

```bash
sed -n '1,240p' .agents/rules/environment.md
```

The key rule: do not run CUDA, PyTorch GPU, embeddings, transformers, or substantial model inference directly on the login node. Use Slurm.

## GPU Workflow

For GPU/model work, prefer the Slurm GPU partition:

```bash
srun -p shared_a6000 --gres=gpu:1 --cpus-per-task=4 --mem=8G --time=01:00:00 .venv/bin/python -u your_script.py
```

The local `.venv` may report CUDA unavailable on the login node, but it can see the GPU when launched through `srun`. Verify with:

```bash
srun -p shared_a6000 --gres=gpu:1 --cpus-per-task=4 --mem=8G --time=00:10:00 .venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected GPU-node result:

```text
cuda_available True
device NVIDIA RTX A6000
```

The `.agents` rule also documents this container helper:

```bash
srun -p shared_a6000 --gres=gpu:1 --cpus-per-task=4 --mem=8G --time=01:00:00 ./scripts/run_gpu_container.sh python your_script.py
```

## Genre Analysis Plot Regeneration

The maintained genre workflow now runs through thin pipeline scripts and immutable run directories. First build or refresh the canonical BERT interaction table on a GPU node:

```bash
srun -p shared_a6000 --gres=gpu:1 --cpus-per-task=4 --mem=8G --time=01:00:00 .venv/bin/python -u pipelines/build_interactions.py --config configs/experiments/genre-analysis.yaml
```

Then run the CPU genre analysis:

```bash
.venv/bin/python -u pipelines/analyze_genres.py --config configs/experiments/genre-analysis.yaml
```

Expected maintained outputs are immutable run directories:

```text
results/runs/<timestamp>-build-interactions-<config-hash>/
results/runs/<timestamp>-analyze-genres-<config-hash>/
```

For local convenience, the current BERT interaction table may also be copied to:

```text
data/processed/speech_interactions_bert.parquet
data/processed/speeches_bert.parquet
```

The old flat `output/` files are preserved as migration baselines only; do not treat them as the active interface.

## Slurm Checks

Check existing jobs before launching GPU work:

```bash
squeue -u shapirma
```

After a run, check again to make sure no leftover jobs remain.

## Process Lessons

- Read `.agents/` first. It contains project-specific environment constraints that are easy to miss.
- Do not trust direct login-node CUDA probes for this repo. The correct question is whether CUDA works inside a Slurm allocation.
- If a model download/cache access fails in the sandbox or through a proxy, do not keep retrying on the login node. Move to the documented Slurm path.
- If a Slurm/container run fails due to missing Python packages, test the project `.venv` under `srun` before installing anything.
