"""Stylometric PCA plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_pca(records: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for genre, group in records.groupby("genre"):
        ax.scatter(group["pca1"], group["pca2"], label=genre)
    ax.set(xlabel="PCA1", ylabel="PCA2", title="Stylometric PCA")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
