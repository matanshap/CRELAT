"""Pure stylometric feature extraction."""

from __future__ import annotations

import re
from collections import Counter
from typing import Mapping

import numpy as np


def clean_folger_text(text: str) -> str:
    text = re.sub(r"(?m)^\s*(ACT|SCENE)\s+[IVXLC\d.]+.*$", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_stylometry(text: str) -> Mapping[str, float]:
    cleaned = clean_folger_text(text)
    words = re.findall(r"[A-Za-z']+", cleaned.casefold())
    sentences = [part for part in re.split(r"[.!?]+", cleaned) if part.strip()]
    if not words:
        raise ValueError("Cannot compute stylometry for empty text")
    counts = Counter(words)
    lengths = np.asarray([len(word) for word in words], dtype=float)
    sentence_lengths = np.asarray(
        [len(re.findall(r"[A-Za-z']+", sentence)) for sentence in sentences], dtype=float
    )
    return {
        "word_count": float(len(words)),
        "vocabulary_size": float(len(counts)),
        "type_token_ratio": float(len(counts) / len(words)),
        "hapax_ratio": float(sum(value == 1 for value in counts.values()) / len(words)),
        "mean_word_length": float(lengths.mean()),
        "word_length_std": float(lengths.std()),
        "mean_sentence_length": float(sentence_lengths.mean()) if sentence_lengths.size else 0.0,
        "sentence_length_std": float(sentence_lengths.std()) if sentence_lengths.size else 0.0,
        "comma_rate": float(cleaned.count(",") / len(words)),
        "semicolon_rate": float(cleaned.count(";") / len(words)),
        "question_rate": float(cleaned.count("?") / len(words)),
        "exclamation_rate": float(cleaned.count("!") / len(words)),
    }
