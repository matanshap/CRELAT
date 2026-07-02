"""Statistical helpers for speech-level analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a one-dimensional array in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


LENGTH_METRICS = (
    "length_difference",
    "length_sum",
    "length_min",
    "length_max",
)


def cluster_bootstrap_correlation(
    frame: pd.DataFrame,
    x_column: str,
    *,
    cluster_column: str = "scene_id",
    iterations: int = 5000,
    seed: int = 20260621,
) -> tuple[float, float]:
    groups = [group.index.to_numpy() for _, group in frame.groupby(cluster_column)]
    if len(groups) < 2:
        raise ValueError("Cluster bootstrap requires at least two clusters")
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[group_index] for group_index in sampled])
        sample = frame.loc[rows]
        estimates[index] = stats.spearmanr(
            sample[x_column], sample["cosine_similarity"]
        ).statistic
    return tuple(float(value) for value in np.nanpercentile(estimates, [2.5, 97.5]))


def length_correlation_results(frame: pd.DataFrame, iterations: int = 5000) -> pd.DataFrame:
    rows = []
    for offset, column in enumerate(LENGTH_METRICS):
        spearman = stats.spearmanr(frame[column], frame["cosine_similarity"])
        pearson = stats.pearsonr(np.log1p(frame[column]), frame["cosine_similarity"])
        low, high = cluster_bootstrap_correlation(
            frame, column, iterations=iterations, seed=20260621 + offset
        )
        rows.append(
            {
                "measure": column,
                "n": len(frame),
                "spearman_rho": spearman.statistic,
                "spearman_ci_low": low,
                "spearman_ci_high": high,
                "spearman_p": spearman.pvalue,
                "pearson_r_log1p": pearson.statistic,
                "pearson_p": pearson.pvalue,
            }
        )
    result = pd.DataFrame(rows)
    result["spearman_q"] = benjamini_hochberg(result["spearman_p"].to_numpy())
    result["pearson_q"] = benjamini_hochberg(result["pearson_p"].to_numpy())
    return result


def cluster_robust_length_regression(frame: pd.DataFrame) -> pd.DataFrame:
    groups = frame["scene_id"].to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Clustered regression requires at least two scenes")
    log_sum = np.log1p(frame["length_sum"].to_numpy(dtype=float))
    normalized_difference = (
        frame["length_difference"] / frame["length_max"].replace(0, np.nan)
    ).fillna(0).to_numpy(dtype=float)

    def zscore(values: np.ndarray) -> np.ndarray:
        deviation = values.std(ddof=0)
        if deviation == 0:
            raise ValueError("Regression predictor has zero variance")
        return (values - values.mean()) / deviation

    design = np.column_stack([np.ones(len(frame)), zscore(log_sum), zscore(normalized_difference)])
    outcome = frame["cosine_similarity"].to_numpy(dtype=float)
    inverse = np.linalg.inv(design.T @ design)
    coefficients = inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    meat = np.zeros((design.shape[1], design.shape[1]))
    for group in unique_groups:
        score = design[groups == group].T @ residuals[groups == group]
        meat += np.outer(score, score)
    correction = len(unique_groups) / (len(unique_groups) - 1) * (
        (len(frame) - 1) / (len(frame) - design.shape[1])
    )
    covariance = correction * inverse @ meat @ inverse
    errors = np.sqrt(np.diag(covariance))
    degrees = len(unique_groups) - 1
    t_values = coefficients / errors
    p_values = 2 * stats.t.sf(np.abs(t_values), degrees)
    return pd.DataFrame(
        {
            "term": ["intercept", "log_length_sum_z", "normalized_difference_z"],
            "coefficient": coefficients,
            "clustered_se": errors,
            "t": t_values,
            "p": p_values,
            "clusters": len(unique_groups),
        }
    )
