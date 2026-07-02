#!/usr/bin/env python3
"""
Standalone: benchmark many classifiers (LOOCV) on CRELAT genre features, then plot the winner.

Reads genre_analysis_data.json only (no other project imports).
Run from repo root:  python3 src/plot_genre_loocv.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC, SVC

DEFAULT_FEATURES = [
    "y_mean",
    "y_iqr",
    "x_gini",
    "x_top1_frac",
    "pearson_r",
    "x_mean",
]


def load_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for r in raw:
        for k, v in r.items():
            if v is None and k not in ("title", "genre", "period"):
                r[k] = np.nan
    return raw


def build_matrix(recs: list[dict], cols: list[str]) -> np.ndarray:
    return np.array([[float(r[k]) for k in cols] for r in recs], dtype=float)


@dataclass
class BenchmarkRow:
    name: str
    feature_tag: str
    macro_f1: float
    weighted_f1: float
    accuracy: float
    error: str | None = None


def _pipe_scaled(clf) -> Pipeline:
    return Pipeline([("sc", StandardScaler()), ("clf", clf)])


def candidate_estimators() -> list[tuple[str, object]]:
    """(short_name, sklearn estimator or Pipeline)."""
    out: list[tuple[str, object]] = []

    for C in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        out.append(
            (
                f"LR C={C}",
                _pipe_scaled(
                    LogisticRegression(max_iter=10000, solver="lbfgs", C=C),
                ),
            )
        )

    out.append(("LDA", _pipe_scaled(LinearDiscriminantAnalysis())))
    for sh in ("auto", 0.2, 0.5, 0.8):
        out.append(
            (
                f"LDA eigen shrink={sh}",
                _pipe_scaled(
                    LinearDiscriminantAnalysis(solver="eigen", shrinkage=sh),
                ),
            )
        )

    for reg in (0.2, 0.1, 0.03):
        out.append(
            (
                f"QDA reg={reg}",
                _pipe_scaled(QuadraticDiscriminantAnalysis(reg_param=reg)),
            )
        )

    for C in (0.5, 1.0, 2.0):
        out.append(
            (
                f"LinearSVC C={C}",
                _pipe_scaled(
                    LinearSVC(C=C, max_iter=20000, dual=False, random_state=0),
                ),
            )
        )

    for C in (1.0, 2.0):
        out.append(
            (
                f"SVM RBF C={C} γ=scale",
                _pipe_scaled(SVC(kernel="rbf", C=C, gamma="scale", random_state=0)),
            )
        )

    out.append(("GaussianNB + scale", _pipe_scaled(GaussianNB())))
    for k in (3, 5, 7):
        out.append((f"KNN k={k}", _pipe_scaled(KNeighborsClassifier(n_neighbors=k))))

    for a in (1.0, 5.0, 15.0):
        out.append(
            (f"RidgeClf α={a}", _pipe_scaled(RidgeClassifier(alpha=a, random_state=0))),
        )

    out.append(
        (
            "SGD log α=1e-3",
            _pipe_scaled(
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-3,
                    max_iter=5000,
                    random_state=0,
                    tol=1e-4,
                ),
            ),
        )
    )

    for depth, ms, n_est in (
        (3, 2, 200),
        (4, 1, 300),
        (5, 1, 300),
        (None, 1, 300),
    ):
        dlabel = "full" if depth is None else str(depth)
        out.append(
            (
                f"RF depth={dlabel} leaf≥{ms}",
                RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_leaf=ms,
                    random_state=0,
                    n_jobs=-1,
                ),
            )
        )

    for depth in (3, 5):
        out.append(
            (
                f"ExtraTrees d={depth}",
                ExtraTreesClassifier(
                    n_estimators=400,
                    max_depth=depth,
                    min_samples_leaf=2,
                    random_state=0,
                    n_jobs=-1,
                ),
            )
        )

    out.append(
        (
            "GradBoost d=2",
            GradientBoostingClassifier(
                max_depth=2,
                n_estimators=150,
                learning_rate=0.08,
                min_samples_leaf=2,
                random_state=0,
            ),
        )
    )
    out.append(
        (
            "GradBoost d=3",
            GradientBoostingClassifier(
                max_depth=3,
                n_estimators=120,
                learning_rate=0.06,
                min_samples_leaf=2,
                random_state=0,
            ),
        )
    )

    try:
        out.append(
            (
                "HistGB d=2",
                HistGradientBoostingClassifier(
                    max_depth=2,
                    max_iter=200,
                    learning_rate=0.06,
                    min_samples_leaf=3,
                    random_state=0,
                ),
            )
        )
        out.append(
            (
                "HistGB d=3",
                HistGradientBoostingClassifier(
                    max_depth=3,
                    max_iter=200,
                    learning_rate=0.05,
                    min_samples_leaf=2,
                    random_state=0,
                ),
            )
        )
    except Exception:
        pass

    out.append(
        (
            "AdaBoost depth1",
            AdaBoostClassifier(
                estimator=RandomForestClassifier(
                    max_depth=1,
                    n_estimators=30,
                    random_state=0,
                    n_jobs=-1,
                ),
                n_estimators=80,
                learning_rate=0.6,
                random_state=0,
            ),
        )
    )

    for C in (0.5, 1.0):
        out.append(
            (
                f"Poly2+LR C={C}",
                Pipeline(
                    [
                        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                        ("sc", StandardScaler()),
                        (
                            "lr",
                            LogisticRegression(max_iter=10000, solver="lbfgs", C=C),
                        ),
                    ]
                ),
            )
        )

    return out


def vote_preset_pipelines() -> dict[str, list]:
    lr05 = _pipe_scaled(LogisticRegression(max_iter=10000, solver="lbfgs", C=0.5))
    lr1 = _pipe_scaled(LogisticRegression(max_iter=10000, solver="lbfgs", C=1.0))
    hist2 = HistGradientBoostingClassifier(
        max_depth=2,
        max_iter=220,
        learning_rate=0.06,
        min_samples_leaf=3,
        random_state=0,
    )
    hist3 = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=200,
        learning_rate=0.05,
        min_samples_leaf=2,
        random_state=0,
    )
    et5 = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=2,
        random_state=0,
        n_jobs=-1,
    )
    rf4 = RandomForestClassifier(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=1,
        random_state=0,
        n_jobs=-1,
    )
    rf3 = RandomForestClassifier(
        n_estimators=400,
        max_depth=3,
        min_samples_leaf=2,
        random_state=0,
        n_jobs=-1,
    )
    return {
        "VOTE: LR0.5+HistGB2+ExtraTrees5": [lr05, hist2, et5],
        "VOTE: LR0.5+HistGB2+RF4": [lr05, hist2, rf4],
        "VOTE: LR1+HistGB2+HistGB3+RF3": [lr1, hist2, hist3, rf3],
        "VOTE: LR0.5+RF4+ExtraTrees5": [lr05, rf4, et5],
    }


def loocv_soft_vote_predict(pipes: list, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Average predict_proba across fitted clones; one held-out sample per fold."""
    loo = LeaveOneOut()
    preds: list[int] = []
    for tr, te in loo.split(X):
        X_tr, X_te = X[tr], X[te]
        y_tr = y[tr]
        acc = None
        for p in pipes:
            m = clone(p)
            m.fit(X_tr, y_tr)
            row = m.predict_proba(X_te)[0].astype(float)
            acc = row if acc is None else acc + row
        acc = acc / len(pipes)
        preds.append(int(np.argmax(acc)))
    return np.array(preds, dtype=int)


def benchmark_votes(X, y, feature_tag: str) -> list[BenchmarkRow]:
    """Hand-picked soft-voting ensembles (often beat a single model on tiny tabular data)."""
    rows: list[BenchmarkRow] = []
    for name, pipes in vote_preset_pipelines().items():
        try:
            pred = loocv_soft_vote_predict(pipes, X, y)
            macro = float(f1_score(y, pred, average="macro"))
            w = float(f1_score(y, pred, average="weighted"))
            acc = float(accuracy_score(y, pred))
            rows.append(BenchmarkRow(name, feature_tag, macro, w, acc, None))
        except Exception as e:
            rows.append(
                BenchmarkRow(name, feature_tag, 0.0, 0.0, 0.0, error=str(e)),
            )
    return rows


def run_loocv_one(estimator, X, y) -> tuple[np.ndarray | None, str | None]:
    loo = LeaveOneOut()
    try:
        pred = cross_val_predict(estimator, X, y, cv=loo)
        return pred, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def benchmark_all(X, y, feature_tag: str) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for name, est in candidate_estimators():
        pred, err = run_loocv_one(est, X, y)
        if err is not None:
            rows.append(
                BenchmarkRow(
                    name, feature_tag, 0.0, 0.0, 0.0, error=err,
                )
            )
            continue
        assert pred is not None
        macro = float(f1_score(y, pred, average="macro"))
        w = float(f1_score(y, pred, average="weighted"))
        acc = float(accuracy_score(y, pred))
        rows.append(BenchmarkRow(name, feature_tag, macro, w, acc, error=None))
    return rows


def plot_confusion(
    y_text: list[str],
    pred_text: list[str],
    labels: list[str],
    title_lines: list[str],
    path: str,
) -> None:
    cm = confusion_matrix(y_text, pred_text, labels=labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="YlOrBr",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Row fraction"},
        ax=ax,
    )
    ax.set_xlabel("Predicted genre (LOOCV)")
    ax.set_ylabel("True genre (Folio-style)")
    ax.set_title("\n".join(title_lines), fontsize=11)
    plt.tight_layout()
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_comparison_bar(rows_ok: list[BenchmarkRow], path: str, top_n: int = 22) -> None:
    rows_ok = sorted(
        rows_ok,
        key=lambda r: (r.macro_f1, r.accuracy),
        reverse=True,
    )[:top_n]
    rows_ok = list(reversed(rows_ok))
    labs = [f"{r.feature_tag}: {r.name}" for r in rows_ok]
    xs = [r.macro_f1 for r in rows_ok]
    fig, ax = plt.subplots(figsize=(10, max(5.0, 0.32 * len(rows_ok))))
    colors = ["#2c7fb8" if "scatter6" in lab else "#7fcdbb" for lab in labs]
    ax.barh(labs, xs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Macro-F1 (leave-one-out)")
    ax.set_title("Genre classification — top methods by macro-F1\n(blue = 6 scatter features only; mint = +composition year)")
    ax.set_xlim(0, 1)
    ax.axvline(xs[-1], color="crimson", ls="--", lw=1, alpha=0.7)
    plt.tight_layout()
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser(
        description="Benchmark classifiers (LOOCV) and plot best confusion + comparison.",
    )
    ap.add_argument(
        "--data",
        default=os.path.join(root, "output", "genre_analysis_data.json"),
        help="Path to genre_analysis_data.json",
    )
    ap.add_argument(
        "--out-confusion",
        default=os.path.join(root, "output", "genre_loocv_confusion.svg"),
        help="Best-model confusion matrix",
    )
    ap.add_argument(
        "--out-compare",
        default=os.path.join(root, "output", "genre_loocv_method_comparison.svg"),
        help="Bar chart of top methods",
    )
    ap.add_argument(
        "--out-json",
        default=os.path.join(root, "output", "genre_loocv_benchmark.json"),
        help="Full benchmark table as JSON",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.data):
        print(f"Data file not found: {args.data}", file=sys.stderr)
        return 1

    records = load_records(args.data)
    scatter_cols = list(DEFAULT_FEATURES)
    recs_scatter = [
        r
        for r in records
        if "genre" in r
        and all(r.get(k) is not None and np.isfinite(float(r[k])) for k in scatter_cols)
    ]
    recs_year = [
        r
        for r in recs_scatter
        if r.get("year") is not None and np.isfinite(float(r["year"]))
    ]
    if len(recs_scatter) < 3:
        print("Too few complete records.", file=sys.stderr)
        return 1

    X6 = build_matrix(recs_scatter, scatter_cols)
    y_text = [r["genre"] for r in recs_scatter]
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    labels = list(le.classes_)

    X7 = build_matrix(recs_year, scatter_cols + ["year"])
    y_text_y = [r["genre"] for r in recs_year]
    le_y = LabelEncoder()
    y_y = le_y.fit_transform(y_text_y)

    all_rows: list[BenchmarkRow] = []
    all_rows.extend(benchmark_all(X6, y, "scatter6"))
    all_rows.extend(benchmark_all(X7, y_y, "scatter6+year"))
    all_rows.extend(benchmark_votes(X6, y, "scatter6"))
    all_rows.extend(benchmark_votes(X7, y_y, "scatter6+year"))

    ok = [r for r in all_rows if r.error is None]
    failed = [r for r in all_rows if r.error is not None]
    if not ok:
        print("All models failed.", file=sys.stderr)
        return 1

    best = max(ok, key=lambda r: (r.macro_f1, r.accuracy, r.weighted_f1))
    ok6 = [r for r in ok if r.feature_tag == "scatter6"]
    best6 = max(ok6, key=lambda r: (r.macro_f1, r.accuracy, r.weighted_f1)) if ok6 else None

    recs_best = recs_year if best.feature_tag == "scatter6+year" else recs_scatter
    cols_best = scatter_cols + (["year"] if best.feature_tag == "scatter6+year" else [])
    X_best = build_matrix(recs_best, cols_best)
    y_best_text = [r["genre"] for r in recs_best]
    le_b = LabelEncoder()
    y_best = le_b.fit_transform(y_best_text)
    lab_best = list(le_b.classes_)

    votes_map = vote_preset_pipelines()
    if best.name in votes_map:
        pred_idx = loocv_soft_vote_predict(votes_map[best.name], X_best, y_best)
    else:
        winner_est = None
        for name, est in candidate_estimators():
            if name == best.name:
                winner_est = est
                break
        if winner_est is None:
            print("Internal error: could not resolve winning estimator.", file=sys.stderr)
            return 1
        pred_idx, err = run_loocv_one(winner_est, X_best, y_best)
        if err or pred_idx is None:
            print(f"Winner re-run failed: {err}", file=sys.stderr)
            return 1

    pred_text = le_b.inverse_transform(pred_idx)
    acc = float(accuracy_score(y_best, pred_idx))
    macro = float(f1_score(y_best, pred_idx, average="macro"))

    title = [
        "Best LOOCV model (many singles + soft-vote; LDA/QDA/SVM/trees/boosting/poly)",
        f"{best.feature_tag} · {best.name}",
        f"n={len(recs_best)} · accuracy={acc:.2f} · macro-F1={macro:.2f}",
    ]
    plot_confusion(y_best_text, pred_text, lab_best, title, args.out_confusion)

    plot_comparison_bar(ok, args.out_compare, top_n=28)

    json_rows = [
        {
            "feature_set": r.feature_tag,
            "model": r.name,
            "macro_f1": r.macro_f1,
            "weighted_f1": r.weighted_f1,
            "accuracy": r.accuracy,
            "error": r.error,
        }
        for r in sorted(all_rows, key=lambda r: (-(r.macro_f1 if not r.error else -1), r.name))
    ]
    best_overall_obj = {
        "feature_set": best.feature_tag,
        "model": best.name,
        "macro_f1": best.macro_f1,
        "accuracy": best.accuracy,
        "n_plays": len(recs_best),
    }
    payload = {
        "best": best_overall_obj,
        "best_overall": best_overall_obj,
        "best_scatter6_only": (
            {
                "model": best6.name,
                "macro_f1": best6.macro_f1,
                "accuracy": best6.accuracy,
            }
            if best6
            else None
        ),
        "note": (
            "Adding composition year partly encodes period, not pure 'text structure' genre. "
            "Compare best_overall vs best_scatter6_only."
        ),
        "rows": json_rows,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    n_single = len(candidate_estimators())
    n_vote = len(vote_preset_pipelines())
    print(
        f"Benchmarked {n_single} single estimators + {n_vote} soft-vote combos × 2 feature sets "
        f"({len(ok)} ok, {len(failed)} failed/skipped).",
    )
    if best6:
        print(
            f"BEST (6 scatter features only, no year): {best6.name}  "
            f"macro-F1={best6.macro_f1:.3f}  acc={best6.accuracy:.3f}",
        )
    print(f"BEST OVERALL: {best.feature_tag} / {best.name}")
    print(f"  macro-F1={best.macro_f1:.3f}  accuracy={best.accuracy:.3f}  n={len(recs_best)}")
    print(f"  confusion → {args.out_confusion}")
    print(f"  comparison chart → {args.out_compare}")
    print(f"  full table → {args.out_json}")
    if failed:
        print(f"\n({len(failed)} configs failed — see JSON 'error' fields.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
