"""Immutable run-directory creation and provenance capture."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from crelat.config import normalized_yaml
from crelat.domain.experiment import ExperimentManifest
from crelat.paths import RESULTS_DIR


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class RunContext:
    def __init__(
        self,
        pipeline: str,
        config: Mapping[str, Any],
        *,
        run_root: Union[str, Path] = RESULTS_DIR,
        force: bool = False,
    ) -> None:
        config_text = normalized_yaml(config)
        config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()[:12]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{timestamp}-{pipeline}-{config_hash}"
        self.path = Path(run_root) / self.run_id
        if self.path.exists() and not force:
            raise FileExistsError(f"Run already exists: {self.path}")
        for directory in ("logs", "data", "tables", "figures"):
            (self.path / directory).mkdir(parents=True, exist_ok=force)
        (self.path / "config.resolved.yaml").write_text(config_text, encoding="utf-8")
        commit = _git(["rev-parse", "HEAD"])
        status = _git(["status", "--porcelain"])
        self.manifest = ExperimentManifest(
            run_id=self.run_id,
            pipeline=pipeline,
            config_hash=config_hash,
            started_at=datetime.now(timezone.utc).isoformat(),
            git_commit=commit,
            git_dirty=bool(status),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            device=os.environ.get("CUDA_VISIBLE_DEVICES", "cpu"),
            metadata={"python": sys.version.split()[0], "platform": platform.platform()},
        )
        (self.path / "environment.txt").write_text(
            f"python={sys.version}\nplatform={platform.platform()}\n",
            encoding="utf-8",
        )
        self._write_manifest()

    def register_input(self, name: str, path: Union[str, Path]) -> None:
        values = dict(self.manifest.inputs)
        values[name] = sha256_file(path)
        self.manifest.inputs = values
        self._write_manifest()

    def complete(self, metadata: Optional[Mapping[str, Any]] = None) -> None:
        artifacts = {}
        for path in sorted(self.path.rglob("*")):
            if path.is_file() and path.name not in {"manifest.json"}:
                artifacts[str(path.relative_to(self.path))] = sha256_file(path)
        self.manifest.artifacts = artifacts
        self.manifest.status = "complete"
        self.manifest.completed_at = datetime.now(timezone.utc).isoformat()
        if metadata:
            self.manifest.metadata = {**self.manifest.metadata, **dict(metadata)}
        self._write_manifest()

    def fail(self, error: BaseException) -> None:
        self.manifest.status = "failed"
        self.manifest.completed_at = datetime.now(timezone.utc).isoformat()
        self.manifest.error = f"{type(error).__name__}: {error}"
        self._write_manifest()

    def _write_manifest(self) -> None:
        (self.path / "manifest.json").write_text(
            json.dumps(asdict(self.manifest), indent=2, sort_keys=True), encoding="utf-8"
        )
