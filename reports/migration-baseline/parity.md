# Migration Parity Record

## Preserved artifacts

Selected active output hashes in `manifest.yaml` were rechecked after restructuring and remained unchanged. Existing tracked output deletions were not restored.

## Antony and Cleopatra BERT

- Baseline and new pipeline: 1,136 consecutive-speech interactions across 40 scenes.
- All `text1` and `text2` values match exactly after correcting TEI whitespace handling.
- Maximum absolute cosine difference: `4.172325134277344e-07`.
- Cosine arrays pass `numpy.allclose(atol=1e-6, rtol=1e-6)`.

## Speech-length statistics

- New pipeline: 1,136 interactions.
- Spearman statistics, confidence intervals, p-values, and adjusted q-values match the preserved baseline exactly.
- Maximum Pearson correlation difference: `1.621727e-08`.
- Maximum Pearson p/q difference: `4.156579e-07`.

## Full corpus

- BERT interaction run: 37 plays, 30,257 interactions, 729 scenes.
- Stylometry run: 37 plays with PCA scores, explained variance, tables, and figure.
- Genre analysis reproduces interaction-count features exactly.
- The cached `genre_analysis_data_bert.json` is dated June 1, while the preserved interaction table is dated June 21. Small semantic-feature differences are therefore recorded but not treated as a same-run regression oracle. Future parity fixtures should be generated atomically from one immutable run.

Validated immutable runs include:

- `20260622T125117Z-build-interactions-ef80e4493baf`
- `20260622T125302Z-analyze-genres-ef80e4493baf`
- `20260622T125539Z-analyze-stylometry-a12757499b7c`
- `20260622T125413Z-analyze-speech-similarity-aea273bb21a2`
- `20260622T125554Z-compare-representations-a1e1aa6f1bb2`
