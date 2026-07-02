"""Genre chronology plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_genre_chronology(features: pd.DataFrame, feature: str, output: Path) -> None:
    rows = features.loc[features["feature"] == feature]
    if rows.empty:
        raise ValueError(f"No rows for feature {feature}")
    colors = {"comedy": "C2", "history": "C0", "tragedy": "C3"}
    fig, ax = plt.subplots(figsize=(11, 7))
    for genre, group in rows.groupby("genre"):
        group = group.sort_values("year")
        ax.scatter(group["year"], group["value"], label=genre, color=colors.get(genre))
        ax.plot(group["year"], group["value"], alpha=0.25, color=colors.get(genre))
    ax.set(title=feature.replace("_", " ").title(), xlabel="Approximate year", ylabel=feature)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
