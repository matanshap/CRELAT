# CRELAT

CRELAT is a reproducible research toolkit for computational analysis of character relationships in Shakespeare's plays. The maintained system parses Folger TEI, embeds speeches, builds consecutive-speech interactions, computes semantic and stylometric features, and writes immutable experiment runs.

The historical Book/Corpus implementation remains available under `legacy/` but is unsupported and is not imported by `crelat`.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Optional model and researcher-corpus support:

```bash
.venv/bin/python -m pip install -e '.[transformers,research]'
```

## Data and configuration

- `data/catalog/plays.yaml` is the only maintained play catalog.
- `data/raw/folger/` contains immutable Folger XML and text.
- `configs/models/` records model identities and sequence limits.
- `configs/experiments/` records analysis choices and seeds.

## Pipelines

CPU stylometry:

```bash
.venv/bin/python pipelines/analyze_stylometry.py \
  --config configs/experiments/stylometry.yaml
```

GPU interaction extraction must run through Slurm:

```bash
srun -p shared_a6000 --gres=gpu:1 --cpus-per-task=4 --mem=8G \
  --time=01:00:00 ./scripts/run_gpu_container.sh python \
  pipelines/build_interactions.py \
  --config configs/experiments/genre-analysis.yaml
```

Every pipeline creates `results/runs/<timestamp>-<pipeline>-<config-hash>/` with resolved configuration, provenance, tables, figures, logs, and checksums. Pipelines never overwrite `reports/`.

## Researcher lens

The repository includes an evidence-grounded research-question skill and an Andrew Piper pilot archive under `research/`. Local books and PDFs are intentionally excluded from Git.
Researcher-lens sessions should also read `docs/research-direction.md` so questions are judged against CRELAT's current project direction rather than against the external researcher alone.

```bash
./scripts/install_researcher_lens.sh
```

The lens reconstructs documented positions, marks speculation, consults criticism, and cites source pages. It does not impersonate the researcher.

## Tests

```bash
MPLCONFIGDIR=/tmp/crelat-mpl .venv/bin/python -m pytest
```

See [architecture](docs/architecture.md) and [HPC execution](docs/hpc.md).
