# CRELAT Research Direction

Use this file as the project compass for researcher-lens sessions. When asking a
researcher lens to generate, scrutinize, or revise a question, the lens should
read this file and judge the response against the direction below.

## Core Aim

CRELAT studies character relationships in Shakespeare through computational
evidence. The current maintained system emphasizes Folger TEI parsing,
speech-level embeddings, consecutive-speech interactions, semantic and
stylometric features, and reproducible experiment runs.

## Current Commitments

- Treat computational outputs as evidence for literary interpretation, not as
  stand-alone scores.
- Prefer questions that connect model behavior to character relations, genre,
  chronology, dramatic structure, or interpretive stakes.
- Keep methods reproducible through explicit configs, immutable run directories,
  provenance, checksums, and curated reports.
- Compare representations and measurements rather than relying on one model or
  one metric as the authority.
- Mark where the corpus, model, feature design, or genre labels constrain the
  claim.

## Active Materials

- Primary corpus: Folger Shakespeare play texts and metadata in `data/`.
- Maintained outputs: immutable pipeline runs in `results/runs/`.
- Curated artifacts: deliberately selected reports in `reports/`.
- Researcher lenses: evidence bases under `research/researchers/`.

## Strong Questions Should

- Name the literary object, unit of analysis, comparison, and interpretive
  contribution.
- Say what would change in our understanding of character relationships if the
  result holds.
- Include plausible controls for frequency, length, genre, chronology, speaker
  distribution, and model effects.
- Be feasible with the maintained pipeline surface, or clearly state what new
  data or method would be required.

## Open Direction Notes

Use this section for team decisions as they become clearer.

- Primary target paper or venue:
- Current strongest claim:
- Current weakest assumption:
- Near-term empirical priority:
- Questions to avoid:
