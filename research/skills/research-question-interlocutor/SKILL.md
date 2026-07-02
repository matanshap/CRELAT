---
name: research-question-interlocutor
description: Develop and scrutinize research questions through evidence-grounded researcher lenses. Use when a user asks to think with, against, compare, or reconstruct a named researcher's ideas; connect a project to that researcher's books and papers; evaluate novelty, operationalization, evidence, confounds, or feasibility; or create a documented research-question session from a local researcher corpus.
---

# Research Question Interlocutor

Use a researcher folder as an evidence base, not as a personality prompt.

## Required workflow

0. Read `docs/research-direction.md` when it exists. Treat it as the user's
   project compass, not as evidence about the external researcher. If it is
   missing or too sparse for the question, ask for or infer a brief working
   direction and label that inference.
1. Locate the requested folder under `research/researchers/`.
2. Read `profile.yaml`, `lens.yaml`, `source_manifest.json`, and `timeline.md`.
3. Follow [researcher-folder-protocol.md](references/researcher-folder-protocol.md).
4. Search indexed source text with `scripts/search_corpus.py`. If no index exists, report the gap; do not substitute summaries for source evidence.
5. Inspect the original extracted pages behind decisive search results.
6. Consult `reception/` before making broad evaluative claims.
7. Apply [evidence-policy.md](references/evidence-policy.md) to every researcher-specific claim.
8. Generate or assess questions with [question-rubric.md](references/question-rubric.md).
9. Return the structure in [output-schema.md](references/output-schema.md).
10. Save serious sessions only when the user requests a file or the active workflow requires one.

When answering, make the researcher's documented commitments interact with the
user's project direction. Do not produce a generic researcher-themed answer if
the question can be narrowed toward CRELAT's corpus, outputs, methods, or
interpretive stakes.

## Lens modes

- **Reconstruct:** Explain a documented position and its development.
- **Generate:** Produce a small set of materially different questions.
- **Scrutinize:** Test a question's concepts, evidence, method, and feasibility.
- **Against:** Identify where the project should reject the researcher's assumptions.
- **Historical:** Compare periods without flattening change over time.
- **Compare:** Put multiple evidence-grounded researcher lenses into disagreement.

Never imitate a researcher in first person. Write “the available evidence suggests” rather than “Researcher X would say.” Mark extrapolation to an unaddressed topic as speculation.
