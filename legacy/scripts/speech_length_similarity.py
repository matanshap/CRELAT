#!/usr/bin/env python3
"""Analyze speech-pair length measures against BERT cosine similarity."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "length_difference": "Absolute length difference",
    "length_sum": "Sum of speech lengths",
    "length_min": "Shorter speech length",
    "length_max": "Longer speech length",
}


def _ascii_apostrophes(value):
    if not isinstance(value, str):
        return value
    return value.replace("\u2018", "'").replace("\u2019", "'")


def benjamini_hochberg(p_values) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def add_length_metrics(
    df: pd.DataFrame,
    tokenizer,
    max_sequence_length: int | None = None,
) -> pd.DataFrame:
    """Tokenize unique speeches and add raw/effective pair-length measures."""
    out = df.copy()
    for column in out.select_dtypes(include="object").columns:
        out[column] = out[column].map(_ascii_apostrophes)

    unique_texts = pd.unique(
        pd.concat([out["text1"], out["text2"]], ignore_index=True)
        .fillna("")
        .astype(str)
    )
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    configured_limit = max_sequence_length or 512
    model_limit = min(int(tokenizer.model_max_length), configured_limit)
    content_limit = model_limit - special_tokens
    raw_lengths = {
        text: len(
            tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        )
        for text in unique_texts
    }

    out["length1_tokens_raw"] = out["text1"].map(raw_lengths)
    out["length2_tokens_raw"] = out["text2"].map(raw_lengths)
    out["length1_tokens"] = out["length1_tokens_raw"].clip(upper=content_limit)
    out["length2_tokens"] = out["length2_tokens_raw"].clip(upper=content_limit)
    out["length1_truncated"] = out["length1_tokens_raw"] > content_limit
    out["length2_truncated"] = out["length2_tokens_raw"] > content_limit
    out["length_difference"] = (
        out["length1_tokens"] - out["length2_tokens"]
    ).abs()
    out["length_sum"] = out["length1_tokens"] + out["length2_tokens"]
    out["length_min"] = out[["length1_tokens", "length2_tokens"]].min(axis=1)
    out["length_max"] = out[["length1_tokens", "length2_tokens"]].max(axis=1)
    out["length_ratio"] = out["length_min"] / out["length_max"].replace(0, np.nan)
    return out


def cluster_bootstrap_correlation(
    df: pd.DataFrame,
    x_column: str,
    cluster_column: str = "scene",
    iterations: int = 5000,
    seed: int = 20260621,
) -> tuple[float, float]:
    """Bootstrap a Spearman CI by resampling scenes with replacement."""
    rng = np.random.default_rng(seed)
    groups = [group.index.to_numpy() for _, group in df.groupby(cluster_column)]
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        sampled_rows = np.concatenate([groups[i] for i in sampled_groups])
        sample = df.loc[sampled_rows]
        estimates[index] = stats.spearmanr(
            sample[x_column], sample["cosine_similarity"]
        ).statistic
    return tuple(np.nanpercentile(estimates, [2.5, 97.5]))


def correlation_results(df: pd.DataFrame, iterations: int) -> pd.DataFrame:
    """Compute correlation effect sizes, clustered CIs, and adjusted p-values."""
    rows = []
    for offset, (column, label) in enumerate(METRICS.items()):
        spearman = stats.spearmanr(df[column], df["cosine_similarity"])
        pearson = stats.pearsonr(
            np.log1p(df[column]), df["cosine_similarity"]
        )
        ci_low, ci_high = cluster_bootstrap_correlation(
            df, column, iterations=iterations, seed=20260621 + offset
        )
        rows.append(
            {
                "measure": column,
                "label": label,
                "n": len(df),
                "spearman_rho": spearman.statistic,
                "spearman_ci_low": ci_low,
                "spearman_ci_high": ci_high,
                "spearman_p": spearman.pvalue,
                "pearson_r_log1p": pearson.statistic,
                "pearson_p": pearson.pvalue,
            }
        )
    result = pd.DataFrame(rows)
    result["spearman_q"] = benjamini_hochberg(result["spearman_p"])
    result["pearson_q"] = benjamini_hochberg(result["pearson_p"])
    return result


def cluster_robust_regression(df: pd.DataFrame) -> pd.DataFrame:
    """OLS with scene-clustered SEs for sum and normalized length difference."""
    log_sum = np.log1p(df["length_sum"].to_numpy(dtype=float))
    normalized_difference = (
        df["length_difference"] / df["length_max"].replace(0, np.nan)
    ).fillna(0).to_numpy(dtype=float)

    def _zscore(values):
        return (values - values.mean()) / values.std(ddof=0)

    x = np.column_stack(
        [np.ones(len(df)), _zscore(log_sum), _zscore(normalized_difference)]
    )
    y = df["cosine_similarity"].to_numpy(dtype=float)
    inverse_xtx = np.linalg.inv(x.T @ x)
    beta = inverse_xtx @ x.T @ y
    residuals = y - x @ beta

    meat = np.zeros((x.shape[1], x.shape[1]))
    groups = df["scene"].to_numpy()
    unique_groups = np.unique(groups)
    for group in unique_groups:
        mask = groups == group
        score = x[mask].T @ residuals[mask]
        meat += np.outer(score, score)
    correction = (
        len(unique_groups) / (len(unique_groups) - 1)
        * (len(df) - 1) / (len(df) - x.shape[1])
    )
    covariance = correction * inverse_xtx @ meat @ inverse_xtx
    standard_errors = np.sqrt(np.diag(covariance))
    degrees_freedom = len(unique_groups) - 1
    critical = stats.t.ppf(0.975, degrees_freedom)
    t_values = beta / standard_errors
    p_values = 2 * stats.t.sf(np.abs(t_values), degrees_freedom)
    r_squared = 1 - np.sum(residuals**2) / np.sum((y - y.mean()) ** 2)

    return pd.DataFrame(
        {
            "term": [
                "intercept",
                "z_log1p_length_sum",
                "z_normalized_length_difference",
            ],
            "coefficient": beta,
            "clustered_se": standard_errors,
            "ci_low": beta - critical * standard_errors,
            "ci_high": beta + critical * standard_errors,
            "t": t_values,
            "p": p_values,
            "scene_clusters": len(unique_groups),
            "n": len(df),
            "r_squared": r_squared,
        }
    )


def _local_average_curve(
    df: pd.DataFrame,
    column: str,
    points: int = 160,
    span: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a continuous tricube-weighted local average without bins."""
    x = df[column].to_numpy(dtype=float)
    y = df["cosine_similarity"].to_numpy(dtype=float)
    grid = np.linspace(x.min(), x.max(), points)
    neighbor_count = max(30, int(np.ceil(span * len(x))))
    smoothed = np.empty_like(grid)

    for index, position in enumerate(grid):
        distances = np.abs(x - position)
        bandwidth = np.partition(distances, neighbor_count - 1)[neighbor_count - 1]
        if bandwidth == 0:
            smoothed[index] = y[distances == 0].mean()
            continue
        scaled = np.minimum(distances / bandwidth, 1.0)
        weights = (1 - scaled**3) ** 3
        smoothed[index] = np.average(y, weights=weights)
    return grid, smoothed


def plot_results(
    df: pd.DataFrame,
    correlations: pd.DataFrame,
    output_path: Path,
    play: str,
    model_label: str = "BERT",
) -> None:
    """Create four simple continuous local-average panels."""
    curves = {
        column: _local_average_curve(df, column)
        for column in METRICS
    }
    all_smoothed = np.concatenate([curve[1] for curve in curves.values()])
    y_span = all_smoothed.max() - all_smoothed.min()
    y_padding = max(0.02, y_span * 0.12)
    y_limits = (
        max(0.0, all_smoothed.min() - y_padding),
        min(1.0, all_smoothed.max() + y_padding),
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
    for ax, (column, label) in zip(axes.flat, METRICS.items()):
        result = correlations.loc[correlations["measure"] == column].iloc[0]
        x_curve, y_curve = curves[column]
        ax.plot(
            x_curve,
            y_curve,
            color="C0",
            linewidth=3,
        )
        ax.set_title(
            f"{label}\n"
            f"Spearman rho = {result.spearman_rho:.3f} "
            f"[{result.spearman_ci_low:.3f}, {result.spearman_ci_high:.3f}]",
            fontweight="bold",
        )
        ax.set_xlabel(f"Effective {model_label} content tokens")
        ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.2)
    axes[0, 0].set_ylabel(f"{model_label} cosine similarity")
    axes[1, 0].set_ylabel(f"{model_label} cosine similarity")
    fig.suptitle(
        f"{play}: Speech Length vs. {model_label} Cosine Similarity",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Lines show continuous local weighted averages (20% nearest-neighbor span); "
        "no length buckets are used.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="output/speech_interactions_bert.csv")
    parser.add_argument("--play", default="Antony and Cleopatra")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument(
        "--output-prefix",
        default="output/speech_length_similarity_antony_and_cleopatra",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    source = pd.read_csv(ROOT / args.input_csv)
    source = source.loc[source["play"].str.casefold() == args.play.casefold()].copy()
    if source.empty:
        raise SystemExit(f"No interactions found for play {args.play!r}.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    analyzed = add_length_metrics(
        source,
        tokenizer,
        max_sequence_length=args.max_sequence_length,
    )
    correlations = correlation_results(analyzed, args.bootstrap_iterations)
    regression = cluster_robust_regression(analyzed)
    model_slug = str(analyzed["model"].iloc[0])
    model_label = args.model_label or {
        "bert": "BERT",
        "macberth": "MacBERTh",
    }.get(model_slug, model_slug)

    prefix = ROOT / args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    analyzed.to_csv(f"{prefix}.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    correlations.to_csv(f"{prefix}_correlations.csv", index=False)
    regression.to_csv(f"{prefix}_regression.csv", index=False)
    plot_results(
        analyzed,
        correlations,
        Path(f"{prefix}.svg"),
        args.play,
        model_label=model_label,
    )

    print(f"Analyzed {len(analyzed)} interactions across {analyzed.scene.nunique()} scenes.")
    print(
        f"Truncated speech cells: "
        f"{int(analyzed.length1_truncated.sum() + analyzed.length2_truncated.sum())}"
    )
    print(correlations.to_string(index=False))
    print(regression.to_string(index=False))
    print(f"Wrote outputs with prefix {prefix}")


if __name__ == "__main__":
    main()
