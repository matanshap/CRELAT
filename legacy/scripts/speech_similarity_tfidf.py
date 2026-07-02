#!/usr/bin/env python3
"""Compare BERT and TF-IDF cosine similarity for adjacent speeches."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


ROOT = Path(__file__).resolve().parents[1]


def _ascii_apostrophes(value):
    if not isinstance(value, str):
        return value
    return value.replace("\u2018", "'").replace("\u2019", "'")


def add_tfidf_similarity(interactions: pd.DataFrame) -> pd.DataFrame:
    """Add pairwise TF-IDF cosine and BERT/TF-IDF percentile disagreement."""
    required = {"text1", "text2", "cosine_similarity"}
    missing = sorted(required - set(interactions.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if interactions.empty:
        raise ValueError("No interactions to analyze.")

    out = interactions.copy()
    for column in out.select_dtypes(include="object").columns:
        out[column] = out[column].map(_ascii_apostrophes)

    text1 = out["text1"].fillna("").astype(str)
    text2 = out["text2"].fillna("").astype(str)
    corpus = pd.unique(pd.concat([text1, text2], ignore_index=True))
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w']+\b",
    )
    vectorizer.fit(corpus)

    vectors1 = vectorizer.transform(text1)
    vectors2 = vectorizer.transform(text2)
    # TfidfVectorizer L2-normalizes rows, so their dot product is cosine.
    tfidf_similarity = np.asarray(vectors1.multiply(vectors2).sum(axis=1)).ravel()

    out = out.rename(columns={"cosine_similarity": "bert_cosine_similarity"})
    out["tfidf_cosine_similarity"] = tfidf_similarity
    out["bert_percentile"] = out["bert_cosine_similarity"].rank(pct=True)
    out["tfidf_percentile"] = out["tfidf_cosine_similarity"].rank(pct=True)
    out["lexical_over_semantic_gap"] = (
        out["tfidf_percentile"] - out["bert_percentile"]
    )
    return out


def plot_similarity_comparison(
    compared: pd.DataFrame,
    output_path: Path,
    play: str,
    labels: int = 10,
) -> None:
    """Plot BERT against TF-IDF cosine and highlight rank disagreements."""
    correlation = compared[
        ["bert_cosine_similarity", "tfidf_cosine_similarity"]
    ].corr(method="spearman").iloc[0, 1]
    max_gap = max(float(compared["lexical_over_semantic_gap"].abs().max()), 0.01)

    fig, ax = plt.subplots(figsize=(13, 9))
    scatter = ax.scatter(
        compared["bert_cosine_similarity"],
        compared["tfidf_cosine_similarity"],
        c=compared["lexical_over_semantic_gap"],
        cmap="coolwarm",
        vmin=-max_gap,
        vmax=max_gap,
        s=34,
        alpha=0.62,
        linewidths=0,
    )

    highlighted = compared.nlargest(labels, "lexical_over_semantic_gap")
    ax.scatter(
        highlighted["bert_cosine_similarity"],
        highlighted["tfidf_cosine_similarity"],
        facecolors="none",
        edgecolors="black",
        s=95,
        linewidths=1.0,
        zorder=3,
    )
    text_labels = []
    for _, row in highlighted.iterrows():
        speaker1 = row["speaker1"].rsplit("_", 1)[0]
        speaker2 = row["speaker2"].rsplit("_", 1)[0]
        text_labels.append(ax.text(
            row["bert_cosine_similarity"],
            row["tfidf_cosine_similarity"],
            f"{row['scene']}: {speaker1} -> {speaker2}",
            fontsize=7,
            alpha=0.9,
        ))
    if adjust_text is not None:
        adjust_text(
            text_labels,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.6),
        )

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    colorbar.set_label("TF-IDF percentile - BERT percentile")
    ax.set_title(
        f"{play}: BERT Semantic vs. TF-IDF Lexical Similarity\n"
        f"Spearman rho = {correlation:.3f}; n = {len(compared):,}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("BERT cosine similarity")
    ax.set_ylabel("TF-IDF cosine similarity")
    ax.grid(True, alpha=0.22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="output/speech_interactions_bert.csv",
        help="Speech-interaction CSV containing BERT cosine similarities.",
    )
    parser.add_argument("--play", default="Antony and Cleopatra")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--output-csv",
        default="output/speech_similarity_bert_tfidf_antony_and_cleopatra.csv",
    )
    parser.add_argument(
        "--disagreements-csv",
        default=(
            "output/speech_similarity_bert_tfidf_disagreements_"
            "antony_and_cleopatra.csv"
        ),
    )
    parser.add_argument(
        "--output-plot",
        default="output/speech_similarity_bert_vs_tfidf_antony_and_cleopatra.svg",
    )
    args = parser.parse_args()

    input_path = ROOT / args.input_csv
    interactions = pd.read_csv(input_path)
    if args.play:
        interactions = interactions.loc[
            interactions["play"].str.casefold() == args.play.casefold()
        ].copy()
    if interactions.empty:
        raise SystemExit(f"No interactions found for play {args.play!r}.")

    compared = add_tfidf_similarity(interactions)
    disagreements = compared.nlargest(args.top, "lexical_over_semantic_gap")

    output_path = ROOT / args.output_csv
    disagreements_path = ROOT / args.disagreements_csv
    plot_path = ROOT / args.output_plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    disagreements_path.parent.mkdir(parents=True, exist_ok=True)
    compared.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    disagreements.to_csv(
        disagreements_path, index=False, quoting=csv.QUOTE_MINIMAL
    )
    plot_similarity_comparison(compared, plot_path, args.play)

    correlation = compared[
        ["bert_cosine_similarity", "tfidf_cosine_similarity"]
    ].corr(method="spearman").iloc[0, 1]
    print(f"Analyzed {len(compared)} interactions for {args.play}.")
    print(f"Spearman correlation (BERT vs TF-IDF): {correlation:.4f}")
    print(f"Wrote all interactions to {output_path}")
    print(f"Wrote top {len(disagreements)} disagreements to {disagreements_path}")
    print(f"Wrote comparison plot to {plot_path}")


if __name__ == "__main__":
    main()
