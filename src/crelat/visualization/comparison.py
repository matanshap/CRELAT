"""Cross-representation comparison figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_isolation_comparison(
    frame: pd.DataFrame, output: Path, left_name: str, right_name: str
) -> None:
    x_column = f"{left_name}_isolation"
    y_column = f"{right_name}_isolation"
    fig, ax = plt.subplots(figsize=(9, 7))
    if "genre" in frame:
        for genre, group in frame.groupby("genre"):
            ax.scatter(group[x_column], group[y_column], label=genre, alpha=0.75)
        ax.legend()
    else:
        ax.scatter(frame[x_column], frame[y_column], alpha=0.75)
    if "title" in frame:
        for row in frame.itertuples():
            ax.annotate(getattr(row, "title"), (getattr(row, x_column), getattr(row, y_column)), fontsize=7)
    ax.set(
        xlabel=f"{left_name} mean absolute isolation",
        ylabel=f"{right_name} mean absolute isolation",
        title="Play isolation across representations",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
