# Intake Workflow

## File Handling

Prefer a user-supplied path. If none is supplied, list `intake/inbox/` and choose the newest non-hidden request file or folder. If there are multiple plausible requests with similar timestamps, ask which one to handle.

Run the inventory helper when the request is a folder or has attachments:

```bash
.venv/bin/python research/skills/crelat-intake-handler/scripts/inventory_intake.py intake/inbox/<request>
```

Use the helper output as an inventory aid, not as a substitute for reading important source files yourself.

## Result Directory

Create one result directory per handled intake item:

```text
results/intake/<YYYYMMDDTHHMMSSZ>-<slug>/
  plan.md
  inventory.json
```

Use UTC timestamps. Build the slug from the request filename or inferred subject. Keep it lowercase and hyphenated.

## Plan Shape

Write `plan.md` with these sections:

- Title
- Intake Summary
- Classification
- Evidence Read
- Proposed Plan
- Tests And Validation
- Assumptions And Blockers

For `unclear` requests, make the plan a clarification brief with the exact missing decisions.

## Extraction Gaps

Record a gap when a file cannot be read locally. Common examples:

- image-only request screenshot
- scanned PDF with no extractable text
- unsupported archive
- audio or video attachment
- binary document format without a local parser

Do not claim to have read a gap file. Ask for text, OCR, transcription, or permission to add tooling only when the gap blocks planning.
