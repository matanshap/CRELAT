---
name: crelat-intake-handler
description: Handle CRELAT intake requests from local files, emails, PDFs, text files, images, or mixed attachment folders. Use when Codex is asked to process new requests placed under intake/inbox, triage emails about creating a new pipeline, updating an existing pipeline, adding a new researcher, or turning messy intake material into a decision-complete plan under results/intake.
---

# CRELAT Intake Handler

Use this skill to turn messy incoming requests into a grounded CRELAT implementation plan. Default to planning first; do not implement the requested work unless the user explicitly asks for implementation after intake handling.

## Required Workflow

1. Read `.agents/rules/environment.md` before running anything nontrivial.
2. Locate the intake item. Prefer a path supplied by the user; otherwise inspect `intake/inbox/` for the newest request-like file or folder.
3. Inventory the intake files. Use `scripts/inventory_intake.py` when useful.
4. Extract what is readable locally:
   - Read `.md`, `.txt`, `.text`, `.rst`, `.yaml`, `.yml`, `.json`, and `.csv` directly.
   - Use local PDF text extraction when available.
   - For images, scanned PDFs, audio, video, archives, or unreadable files, record an explicit extraction gap.
5. Classify the request as `new-pipeline`, `update-pipeline`, `new-researcher`, `mixed`, or `unclear`.
6. Inspect the matching repo surface before planning:
   - Pipelines: `pipelines/`, `configs/experiments/`, `src/crelat/`, `docs/architecture.md`, and `docs/hpc.md`.
   - Researchers: `research/researchers/`, `research/skills/research-question-interlocutor/`, and that skill's researcher-folder protocol.
   - GPU/model/embedding work: route through Slurm; never run substantial model inference on the login node.
7. Write the intake result under `results/intake/<timestamp>-<slug>/plan.md` when the user asks to handle a request or when durable output is implied.

## Planning Rules

- Make the plan decision-complete: goal, classification, discovered evidence, proposed changes, test plan, assumptions, and blockers.
- Distinguish repo facts from inference.
- Quote or summarize only the intake content needed to justify the plan.
- If required request content is unreadable, do not guess. Name the missing file and ask for OCR, transcription, or a text paste.
- Keep raw intake files in `intake/inbox/`; they are private/local by default.
- Treat `results/intake/` as generated output. Avoid committing request-specific plans unless the user explicitly asks for a sanitized artifact.

## References

- Read `references/intake-workflow.md` for the planning output shape and file handling details.
- Read `references/request-types.md` for CRELAT-specific routing for pipeline and researcher requests.
