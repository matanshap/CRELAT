"""Play-level semantic feature analysis."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from crelat.analysis.isolation import mean_absolute_isolation
from crelat.domain.play import PlaySpec
from crelat.features.interactions import compute_pair_features


def compute_genre_features(
    pairs_by_play: dict[str, pd.DataFrame],
    catalog: Iterable[PlaySpec],
    *,
    representation: str,
    y_mean_mode: str = "pair",
) -> pd.DataFrame:
    specs = {play.id: play for play in catalog}
    rows = []
    for play_id, pairs in pairs_by_play.items():
        if play_id not in specs:
            raise KeyError(f"Unknown play_id: {play_id}")
        values = compute_pair_features(pairs, y_mean_mode=y_mean_mode)
        spec = specs[play_id]
        rows.extend(
            {
                "play_id": play_id,
                "title": spec.title,
                "genre": spec.genre,
                "year": spec.year,
                "representation": representation,
                "feature": feature,
                "value": value,
                "transform": "sqrt" if feature.startswith("y_std") or feature.startswith("y_root") else "identity",
            }
            for feature, value in values.items()
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    isolation_rows = []
    for feature, group in frame.groupby("feature"):
        distances = mean_absolute_isolation(group["value"].to_numpy())
        for (_, row), distance in zip(group.iterrows(), distances):
            isolation_rows.append(
                {
                    **row.to_dict(),
                    "feature": f"isolation::{feature}",
                    "value": distance,
                    "transform": "mean_absolute_distance",
                }
            )
    return pd.concat([frame, pd.DataFrame(isolation_rows)], ignore_index=True)
