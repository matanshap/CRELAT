"""Character interaction construction and semantic aggregation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from crelat.domain.interaction import SpeechInteraction
from crelat.domain.play import Play


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError("Embedding arrays must have the same shape")
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = left_norm * right_norm
    values = np.divide(
        np.sum(left * right, axis=1),
        denominator,
        out=np.full(len(left), np.nan, dtype=float),
        where=denominator > 0,
    )
    return values


def build_speech_interactions(
    play: Play,
    embeddings: np.ndarray,
    model_id: str,
) -> pd.DataFrame:
    """Build directed consecutive-speech interactions within each scene."""
    speeches = play.speeches
    if embeddings.ndim != 2 or embeddings.shape[0] != len(speeches):
        raise ValueError("Expected one 2-D embedding row per speech")
    by_id = {speech.id: embeddings[index] for index, speech in enumerate(speeches)}
    records: list[SpeechInteraction] = []
    for scene in play.scenes:
        for first, second in zip(scene.speeches, scene.speeches[1:]):
            if first.speaker_id == second.speaker_id:
                continue
            similarity = cosine_rows(
                np.asarray([by_id[first.id]]), np.asarray([by_id[second.id]])
            )[0]
            if not np.isfinite(similarity):
                continue
            records.append(
                SpeechInteraction(
                    play_id=play.id,
                    play_title=play.title,
                    scene_id=scene.id,
                    speech1_id=first.id,
                    speech2_id=second.id,
                    speaker1=first.speaker_id,
                    speaker2=second.speaker_id,
                    text1=first.text,
                    text2=second.text,
                    cosine_similarity=float(similarity),
                    model_id=model_id,
                )
            )
    return pd.DataFrame([asdict(record) for record in records])


def top_speakers(play: Play, count: int = 8) -> list[str]:
    frequencies = Counter(
        speech.speaker_id for speech in play.speeches if speech.speaker_id != "[UNKNOWN]"
    )
    return [speaker for speaker, _ in frequencies.most_common(count)]


def _undirected_pair(frame: pd.DataFrame) -> pd.Series:
    return frame[["speaker1", "speaker2"]].apply(
        lambda row: "||".join(sorted((str(row.iloc[0]), str(row.iloc[1])))), axis=1
    )


def aggregate_character_pairs(
    interactions: pd.DataFrame,
    *,
    speakers: Optional[Iterable[str]] = None,
    min_interactions: int = 1,
) -> pd.DataFrame:
    frame = interactions.copy()
    if speakers is not None:
        allowed = set(speakers)
        frame = frame[frame["speaker1"].isin(allowed) & frame["speaker2"].isin(allowed)]
    frame = frame[frame["speaker1"] != frame["speaker2"]]
    frame["pair_id"] = _undirected_pair(frame)
    grouped = frame.groupby("pair_id", sort=True)["cosine_similarity"]
    result = grouped.agg(interactions="size", similarity_mean="mean").reset_index()
    variances = grouped.apply(lambda values: float(np.var(values.to_numpy(dtype=float))))
    result["similarity_variance"] = result["pair_id"].map(variances)
    return result.loc[result["interactions"] >= min_interactions].reset_index(drop=True)


def gini(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or np.allclose(array.sum(), 0):
        return 0.0
    if np.any(array < 0):
        raise ValueError("Gini values must be nonnegative")
    array = np.sort(array)
    index = np.arange(1, array.size + 1)
    return float(np.sum((2 * index - array.size - 1) * array) / (array.size * array.sum()))


def compute_pair_features(
    pairs: pd.DataFrame, *, y_mean_mode: str = "pair"
) -> Mapping[str, float]:
    if y_mean_mode not in {"pair", "interaction"}:
        raise ValueError("y_mean_mode must be 'pair' or 'interaction'")
    if len(pairs) < 2:
        raise ValueError("At least two character pairs are required")
    counts = pairs["interactions"].to_numpy(dtype=float)
    means = pairs["similarity_mean"].to_numpy(dtype=float)
    variances = pairs["similarity_variance"].to_numpy(dtype=float)
    weights = counts if y_mean_mode == "interaction" else None
    y_mean = float(np.average(means, weights=weights))
    average_variance = float(np.average(variances, weights=weights))
    average_std = float(np.average(np.sqrt(variances), weights=weights))
    variance_of_averages = float(np.var(means))
    total = counts.sum()
    return {
        "y_mean": y_mean,
        "y_variance_of_averages": variance_of_averages,
        "y_average_of_variances": average_variance,
        "y_std_of_averages": float(np.sqrt(variance_of_averages)),
        "y_average_of_std_devs": average_std,
        "y_root_average_of_variances": float(np.sqrt(average_variance)),
        "y_iqr": float(np.percentile(means, 75) - np.percentile(means, 25)),
        "x_mean": float(np.mean(counts)),
        "x_gini": gini(counts),
        "x_top1_frac": float(counts.max() / total) if total else 0.0,
        "pearson_r": float(np.corrcoef(counts, means)[0, 1]),
    }
