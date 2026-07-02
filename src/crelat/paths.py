"""Project path discovery without process-wide working-directory changes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def project_root() -> Path:
    configured = os.environ.get("CRELAT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = project_root()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = Path(os.environ.get("CRELAT_CACHE_DIR", PROJECT_ROOT / ".cache"))


def resolve_project_path(value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
