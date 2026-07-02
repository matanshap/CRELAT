"""Experiment provenance record."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class ExperimentManifest:
    run_id: str
    pipeline: str
    config_hash: str
    started_at: str
    status: str = "running"
    completed_at: Optional[str] = None
    git_commit: Optional[str] = None
    git_dirty: bool = False
    slurm_job_id: Optional[str] = None
    device: Optional[str] = None
    inputs: Mapping[str, str] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
