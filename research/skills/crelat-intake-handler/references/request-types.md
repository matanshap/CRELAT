# Request Types

## New Pipeline

Classify as `new-pipeline` when the intake asks for a new reproducible analysis, data transformation, figure workflow, experiment, or model-driven processing path.

Plan against the maintained architecture:

- Thin entrypoint under `pipelines/`.
- Reusable behavior under `src/crelat/`.
- Experiment configuration under `configs/experiments/`.
- Immutable outputs under `results/runs/`.
- GPU/model work through Slurm only.

The plan must name the expected inputs, outputs, config keys, run artifacts, and validation path.

## Update Existing Pipeline

Classify as `update-pipeline` when the intake refers to changing behavior, config, outputs, figures, or analysis in an existing pipeline.

Before planning, inspect the current pipeline script, its config, related `src/crelat/` modules, and any baseline report mentioned by the request. Preserve existing run-directory behavior unless the request explicitly changes it.

## New Researcher

Classify as `new-researcher` when the intake asks to add a scholar, corpus, bibliography, lens, or evidence base.

Plan under `research/researchers/<slug>/`, not under `src/crelat/`. Use the `research-question-interlocutor` skill and its researcher-folder protocol. Track source metadata and evidence gaps explicitly. Do not invent source claims from summaries.

## Mixed

Classify as `mixed` when the request combines pipeline work and researcher-corpus work. Split the plan into phases and identify dependencies between them.

## Unclear

Classify as `unclear` when the readable content does not identify a concrete target, expected output, or enough source material. Produce a clarification brief instead of an implementation plan.
