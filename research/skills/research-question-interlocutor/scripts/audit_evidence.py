#!/usr/bin/env python3
"""Validate a session evidence ledger."""

import argparse
import json
from pathlib import Path

LABELS = {"documented", "synthesis", "speculation", "contested", "unknown"}


def audit(payload):
    errors = []
    for index, claim in enumerate(payload.get("evidence_ledger", []), start=1):
        label = claim.get("label")
        if label not in LABELS:
            errors.append(f"claim {index}: invalid label")
        if label in {"documented", "synthesis", "contested"}:
            if not claim.get("work_id"):
                errors.append(f"claim {index}: missing work_id")
            if claim.get("page") is None and not claim.get("section"):
                errors.append(f"claim {index}: missing page or section")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    errors = audit(json.loads(args.session.read_text(encoding="utf-8")))
    if errors:
        raise SystemExit("\n".join(errors))
    print("Evidence ledger valid")


if __name__ == "__main__":
    main()
