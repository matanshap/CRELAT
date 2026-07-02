# CRELAT Architecture

## Data flow

```text
Folger XML -> Play/Scene/Speech -> text embeddings -> speech interactions
                                                -> pair aggregates
                                                -> play features
                                                -> analyses and figures
```

Parsing, model inference, feature extraction, analysis, and visualization are separate layers. Domain and analysis code never chooses output paths. Pipeline scripts perform orchestration and create immutable run directories.

## Package boundaries

- `crelat.domain`: typed records with no I/O or model dependencies.
- `crelat.io`: Folger and tabular formats only.
- `crelat.embeddings`: model loading, batching, pooling, and device policy.
- `crelat.features`: deterministic measurements over parsed or embedded data.
- `crelat.analysis`: statistical and cross-play interpretation.
- `crelat.visualization`: figures from DataFrames.
- `pipelines`: strict configuration loading and run orchestration.

All play joins use stable `play_id` values from `data/catalog/plays.yaml`.

## Results

Run directories are immutable research records. `manifest.json` records configuration hash, Git state, input and artifact checksums, environment, Slurm metadata, status, and timestamps. Selected publication-ready artifacts may be copied deliberately into `reports/`.

## Researcher subsystem

The researcher lens is deliberately outside `src/crelat`. Its skill defines the reasoning procedure; researcher folders supply source metadata, local corpus material, indexes, criticism, and navigation maps. Summaries guide retrieval but cannot serve as primary evidence.
