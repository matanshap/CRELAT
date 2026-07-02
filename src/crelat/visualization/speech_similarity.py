"""Speech-level similarity diagnostics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_length_similarity(records: pd.DataFrame, output: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(records["length_sum"], records["cosine_similarity"], alpha=0.35)
    axes[0].set(xlabel="Combined token length", ylabel="Cosine similarity")
    axes[1].scatter(records["length_difference"], records["cosine_similarity"], alpha=0.35)
    axes[1].set(xlabel="Absolute token-length difference", ylabel="Cosine similarity")
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
