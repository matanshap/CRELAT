#!/usr/bin/env python3
"""Create a play-to-play TF-IDF lexical-similarity heatmap."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from genre_analysis import PLAYS  # noqa: E402
from xmlparser import XMLParser  # noqa: E402


def _ascii_apostrophes(text: str) -> str:
    return text.replace("\u2018", "'").replace("\u2019", "'")


def load_play_documents() -> tuple[list[str], list[str]]:
    """Load one cleaned speech-text document for every configured play."""
    titles = []
    documents = []
    for xml_path, title, _genre, _year in PLAYS:
        parser = XMLParser(str(ROOT / xml_path), options=set())
        parser.parse()
        speeches = [
            speech["text"]
            for scene in parser.characters_speeches
            for speech in scene
            if speech["text"]
        ]
        titles.append(_ascii_apostrophes(title))
        documents.append(_ascii_apostrophes("\n".join(speeches)))
    return titles, documents


def compute_tfidf_similarity(documents: list[str]) -> np.ndarray:
    """Compute word unigram/bigram TF-IDF cosine between documents."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w']+\b",
    )
    vectors = vectorizer.fit_transform(documents)
    return cosine_similarity(vectors)


def plot_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
    """Plot off-diagonal play similarities as a square heatmap."""
    values = matrix.to_numpy()
    mask = np.eye(len(matrix), dtype=bool)
    off_diagonal = values[~mask]

    fig, ax = plt.subplots(figsize=(19, 17))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    image = ax.imshow(
        np.ma.array(values, mask=mask),
        cmap=cmap,
        vmin=float(off_diagonal.min()),
        vmax=float(off_diagonal.max()),
        aspect="equal",
        interpolation="nearest",
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.75, pad=0.02)
    colorbar.set_label("TF-IDF cosine similarity")
    ax.set_title(
        "Shakespeare Plays: Pairwise TF-IDF Lexical Similarity",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Play")
    ax.set_ylabel("Play")
    positions = np.arange(len(matrix))
    ax.set_xticks(positions, matrix.columns, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(positions, matrix.index, fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(matrix), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.15)
    ax.tick_params(which="minor", bottom=False, left=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-plot",
        default="output/play_tfidf_similarity_heatmap.svg",
    )
    parser.add_argument(
        "--output-csv",
        default="output/play_tfidf_similarity_matrix.csv",
    )
    args = parser.parse_args()

    titles, documents = load_play_documents()
    similarities = compute_tfidf_similarity(documents)
    matrix = pd.DataFrame(similarities, index=titles, columns=titles)

    output_csv = ROOT / args.output_csv
    output_plot = ROOT / args.output_plot
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_csv, index_label="play")
    plot_heatmap(matrix, output_plot)

    mask = ~np.eye(len(matrix), dtype=bool)
    pairs = [
        (similarities[i, j], titles[i], titles[j])
        for i in range(len(titles))
        for j in range(i + 1, len(titles))
    ]
    pairs.sort(reverse=True)
    print(f"Compared {len(titles)} plays ({int(mask.sum() / 2)} unique pairs).")
    print(
        "Most similar pair: "
        f"{pairs[0][1]} / {pairs[0][2]} ({pairs[0][0]:.4f})"
    )
    print(f"Wrote similarity matrix to {output_csv}")
    print(f"Wrote heatmap to {output_plot}")


if __name__ == "__main__":
    main()
