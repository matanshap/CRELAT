"""Character-pair interaction plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_pair_scatter(pairs: pd.DataFrame, output: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(pairs["interactions"], pairs["similarity_mean"], alpha=0.7)
    ax.set(title=title, xlabel="Consecutive speech interactions", ylabel="Mean cosine similarity")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
