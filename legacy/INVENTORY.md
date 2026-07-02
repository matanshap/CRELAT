# Permanent Legacy Archive

The files in this directory are retained for provenance and reference. They are unsupported, may depend on obsolete paths and environments, and must not be imported by `src/crelat`.

## `src/`

- `Book.py`, `Corpus.py`, `RunCRELAT_*`: original generic book/corpus workflows.
- `W2V.py`, `Bert_MLM.py`, `OLMo_Embeddings.py`, `gguf_*`: historical model experiments.
- `xmlparser.py`: pre-restructure monolithic Folger parser, model runner, analysis, and plotting code.
- `genre_*`, `stylometry_*`, `character_*`: predecessor analysis scripts whose maintained concepts were extracted into `crelat`.
- `act_entropy_analysis.py`, `llm_predictability.py`, `model_modern.py`: exploratory LLM experiments outside maintained scope.

## Other directories

- `scripts/`: predecessor one-off analysis entry points.
- `notebooks/` and `scratch/`: exploratory notebooks and extraction experiments.
- `data/`: original `Data/` tree and unrelated root-level corpora/logs.
- `requirements/`: historical machine-specific environment dump.

Git commit `51955324aca4f29cecaeab41a001deee56ce8c29` is the migration baseline. User modifications present at migration time were moved intact, not reverted.
