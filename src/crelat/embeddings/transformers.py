"""Hugging Face embedding adapter with HPC execution guardrails."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from crelat.embeddings.base import TextEmbedder

MODEL_REGISTRY: Mapping[str, Mapping[str, Any]] = {
    "bert": {"hf_name": "bert-base-uncased", "pooling": "mean", "max_length": 512},
    "macberth": {
        "hf_name": "emanjavacas/MacBERTh",
        "pooling": "mean",
        "max_length": 512,
    },
    "mpnet": {
        "hf_name": "sentence-transformers/all-mpnet-base-v2",
        "pooling": "mean",
        "max_length": 384,
    },
}


def model_slug(model_id: str) -> str:
    for slug, config in MODEL_REGISTRY.items():
        if model_id in {slug, config["hf_name"]}:
            return slug
    return re.sub(r"[^a-z0-9]+", "-", model_id.casefold()).strip("-")


def _require_slurm() -> None:
    if not os.environ.get("SLURM_JOB_ID") and os.environ.get("CRELAT_ALLOW_LOGIN_MODEL") != "1":
        raise RuntimeError(
            "Transformer inference must run in a Slurm allocation. Use "
            "srun -p shared_a6000 --gres=gpu:1 ./scripts/run_gpu_container.sh python ..."
        )


@dataclass
class TransformerEmbedder(TextEmbedder):
    model_id: str
    batch_size: int = 16
    max_length: Optional[int] = None
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self.output_dimension = 0

    def _load(self) -> None:
        _require_slurm()
        import torch
        from transformers import AutoModel, AutoTokenizer

        config = MODEL_REGISTRY.get(self.model_id, {})
        hf_name = str(config.get("hf_name", self.model_id))
        self.max_length = self.max_length or int(config.get("max_length", 512))
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(hf_name)
        self._model = AutoModel.from_pretrained(hf_name).to(self.device).eval()
        self.output_dimension = int(self._model.config.hidden_size)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            self._load()
        import torch

        rows = []
        assert self._tokenizer is not None and self._model is not None
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self._model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            rows.append(pooled.cpu().numpy())
        return np.vstack(rows) if rows else np.empty((0, self.output_dimension), dtype=float)


def create_embedder(model_id: str, **kwargs: Any) -> TextEmbedder:
    return TransformerEmbedder(model_id=model_id, **kwargs)
