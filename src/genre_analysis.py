#!/usr/bin/env python3
"""
Genre and chronology analysis for Shakespeare play features.

Computes per-play features from the BERT-normalized interaction scatter
(X = co-occurrence count, Y = BERT cosine similarity / interactions),
caches them, and plots chronological trends by genre with polynomial
fitting (degree configurable via --degree).
"""
from __future__ import annotations

import json
import os
import sys
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(__file__))

try:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import PolynomialFeatures
except ImportError:
    RidgeCV = None
    PolynomialFeatures = None

try:
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    DecisionTreeClassifier = None

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

DEFAULT_CHRONOLOGICAL_BY_GENRE_FEATURES = [
    "y_mean", "y_variance_of_averages", "y_average_of_variances",
    "y_iqr", "x_gini", "x_top1_frac", "pearson_r", "x_mean",
    "isolation_xy_mean", "isolation_dy_mean",
]

# Shakespeare plays with genre and approximate year of composition.
# Poems (Son, Ven, Luc, PhT) and TNK (co-authored) excluded.
PLAYS = [
    # Comedies
    ("Data/XML/folger_corpus/Err.xml",  "The Comedy of Errors",       "comedy",  1594),
    ("Data/XML/folger_corpus/TGV.xml",  "The Two Gentlemen of Verona","comedy",  1594),
    ("Data/XML/folger_corpus/Shr.xml",  "The Taming of the Shrew",   "comedy",  1593),
    ("Data/XML/folger_corpus/LLL.xml",  "Love's Labour's Lost",      "comedy",  1595),
    ("Data/XML/folger_corpus/MND.xml",  "A Midsummer Night's Dream", "comedy",  1595),
    ("Data/XML/folger_corpus/MV.xml",   "The Merchant of Venice",    "comedy",  1596),
    ("Data/XML/folger_corpus/Wiv.xml",  "The Merry Wives of Windsor","comedy",  1597),
    ("Data/XML/folger_corpus/Ado.xml",  "Much Ado About Nothing",    "comedy",  1598),
    ("Data/XML/folger_corpus/AYL.xml",  "As You Like It",            "comedy",  1599),
    ("Data/XML/folger_corpus/TN.xml",   "Twelfth Night",             "comedy",  1601),
    ("Data/XML/folger_corpus/AWW.xml",  "All's Well That Ends Well", "comedy",  1602),
    ("Data/XML/folger_corpus/MM.xml",   "Measure for Measure",       "comedy",  1604),
    # Tragedies
    ("Data/XML/folger_corpus/Tit.xml",  "Titus Andronicus",          "tragedy", 1593),
    ("Data/XML/folger_corpus/Rom.xml",  "Romeo and Juliet",          "tragedy", 1595),
    ("Data/XML/folger_corpus/JC.xml",   "Julius Caesar",             "tragedy", 1599),
    ("Data/XML/folger_corpus/Ham.xml",  "Hamlet",                    "tragedy", 1600),
    ("Data/XML/folger_corpus/Oth.xml",  "Othello",                   "tragedy", 1604),
    ("Data/XML/folger_corpus/Lr.xml",   "King Lear",                 "tragedy", 1606),
    ("Data/XML/folger_corpus/Mac.xml",  "Macbeth",                   "tragedy", 1606),
    ("Data/XML/folger_corpus/Ant.xml",  "Antony and Cleopatra",      "tragedy", 1607),
    ("Data/XML/folger_corpus/Cor.xml",  "Coriolanus",                "tragedy", 1608),
    ("Data/XML/folger_corpus/Tim.xml",  "Timon of Athens",           "tragedy", 1608),
    ("Data/XML/folger_corpus/Tro.xml",  "Troilus and Cressida",      "tragedy", 1602),
    # Histories
    ("Data/XML/folger_corpus/1H6.xml",  "Henry VI Part 1",           "history", 1591),
    ("Data/XML/folger_corpus/2H6.xml",  "Henry VI Part 2",           "history", 1591),
    ("Data/XML/folger_corpus/3H6.xml",  "Henry VI Part 3",           "history", 1591),
    ("Data/XML/folger_corpus/R3.xml",   "Richard III",               "history", 1592),
    ("Data/XML/folger_corpus/R2.xml",   "Richard II",                "history", 1595),
    ("Data/XML/folger_corpus/Jn.xml",   "King John",                 "history", 1596),
    ("Data/XML/folger_corpus/1H4.xml",  "Henry IV Part 1",           "history", 1597),
    ("Data/XML/folger_corpus/2H4.xml",  "Henry IV Part 2",           "history", 1598),
    ("Data/XML/folger_corpus/H5.xml",   "Henry V",                   "history", 1599),
    ("Data/XML/folger_corpus/H8.xml",   "Henry VIII",                "history", 1613),
    # Late Romances
    ("Data/XML/folger_corpus/Per.xml",  "Pericles",                  "comedy",  1608),
    ("Data/XML/folger_corpus/Cym.xml",  "Cymbeline",                 "comedy",  1610),
    ("Data/XML/folger_corpus/WT.xml",   "The Winter's Tale",         "comedy",  1611),
    ("Data/XML/folger_corpus/Tmp.xml",  "The Tempest",               "comedy",  1611),
]

PERIODS = {"early": (1589, 1598), "middle": (1599, 1606), "late": (1607, 1614)}

TOP_N = 8
MIN_COOC = 3


# ── helpers ──────────────────────────────────────────────────────

def _get_top_speakers(parser, top_n=TOP_N):
    speech_counts = {char: 0 for char in parser.characters}
    for scene in parser.characters_speeches:
        for speech in scene:
            speaker = speech.get("speaker", "[UNKNOWN]")
            if speaker in speech_counts:
                speech_counts[speaker] += 1
    ranked = sorted(speech_counts.items(), key=lambda x: x[1], reverse=True)
    ranked = [item for item in ranked if item[0] != "[UNKNOWN]"]
    return [char for char, _ in ranked[:top_n]]


def _gini(values):
    """Gini coefficient of a 1-D array."""
    a = np.sort(np.abs(np.asarray(values, dtype=float)))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * a) - (n + 1) * np.sum(a)) / (n * np.sum(a)))


def _extract_scatter_data(parser, characters, min_cooc=MIN_COOC, embedding_type="bert"):
    """
    From a parsed play, extract x_vals (co-occurrence), y_vals
    (cosine similarity / co-occurrence), and y_vars (variance of cosine similarities)
    for all character pairs.
    """
    from collections import defaultdict
    pair_to_similarities = defaultdict(list)
    for interaction in getattr(parser, "speech_interactions", []):
        if interaction.get("model") == embedding_type:
            s1 = interaction["speaker1"]
            s2 = interaction["speaker2"]
            pair = tuple(sorted([s1, s2]))
            pair_to_similarities[pair].append(interaction["cosine_similarity"])

    x_vals, y_vals, y_vars = [], [], []
    for i, c1 in enumerate(characters):
        for c2 in characters[i + 1:]:
            cooc = parser.co_occurrences[c1][c2]
            if cooc < min_cooc:
                continue

            pair = tuple(sorted([c1, c2]))
            sims = pair_to_similarities.get(pair, [])
            if not sims:
                # Fallback to existing logic if speech_interactions is empty/missing
                cosine_map = parser.cosine_similarities.get(embedding_type, {})
                cosim = cosine_map.get(c1, {}).get(c2, np.nan)
                if np.isnan(cosim) or cooc == 0:
                    continue
                pair_mean = float(cosim) / float(cooc)
                pair_var = 0.0
            else:
                pair_mean = float(np.mean(sims))
                pair_var = float(np.var(sims))

            if np.isnan(pair_mean):
                continue
            x_vals.append(float(cooc))
            y_vals.append(pair_mean)
            y_vars.append(pair_var)

    return np.array(x_vals), np.array(y_vals), np.array(y_vars)


def _compute_features(x_vals, y_vals, y_vars, y_mean_mode="pair"):
    """Compute scatter-level features from x, y, and variance arrays."""
    if len(x_vals) < 2:
        return {}
    r, _ = sp_stats.pearsonr(x_vals, y_vals) if len(x_vals) >= 2 else (np.nan, np.nan)
    q75, q25 = np.percentile(y_vals, [75, 25])
    total_x = x_vals.sum()
    top1_frac = float(x_vals.max() / total_x) if total_x > 0 else 0.0

    if y_mean_mode == "interaction":
        y_mean = float(np.average(y_vals, weights=x_vals)) if total_x > 0 else np.nan
        y_average_of_variances = float(np.average(y_vars, weights=x_vals)) if total_x > 0 else np.nan
    else:
        y_mean = float(np.mean(y_vals))
        y_average_of_variances = float(np.mean(y_vars))

    y_variance_of_averages = float(np.var(y_vals))

    return {
        "y_mean": y_mean,
        "y_variance_of_averages": y_variance_of_averages,
        "y_average_of_variances": y_average_of_variances,
        "y_iqr": float(q75 - q25),
        "x_mean": float(np.mean(x_vals)),
        "x_gini": float(_gini(x_vals)),
        "x_top1_frac": top1_frac,
        "pearson_r": float(r),
    }


def _period_for_year(year):
    for name, (lo, hi) in PERIODS.items():
        if lo <= year <= hi:
            return name
    return "unknown"


# ── data I/O ─────────────────────────────────────────────────────

def load_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for r in raw:
        for k, v in r.items():
            if v is None and k not in ("title", "genre", "period"):
                r[k] = np.nan
    return raw


def save_records(records: list[dict], path: str) -> None:
    def _enc(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _enc(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_enc(x) for x in obj]
        return obj
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_enc(records), f, indent=2)


# ── compute pipeline ─────────────────────────────────────────────

def compute_all_records(plays=PLAYS, top_n=TOP_N, min_cooc=MIN_COOC, model="bert", y_mean_mode="pair"):
    """Compute per-play features using the given embedding *model*.

    *model* can be any key from ``MODEL_REGISTRY`` (``"bert"``,
    ``"macberth"``, ``"olmo"``) or any HuggingFace model name.
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from xmlparser import XMLParser, resolve_model
    slug, _, _ = resolve_model(model)
    records = []
    all_interactions = []
    for xml_path, title, genre, year in plays:
        print(f"\n{'='*60}")
        print(f"  {title} ({genre}, {year})  [{model}]")
        print(f"  Mode: y_mean = {y_mean_mode}_average")
        print(f"{'='*60}")
        if not os.path.isfile(xml_path):
            print(f"  ⚠ XML not found: {xml_path} — skipping")
            continue
        try:
            parser = XMLParser(xml_path, options={"co-oc", model})
            parser.parse()
            all_interactions.extend(parser.speech_interactions)
        except Exception as e:
            print(f"  ⚠ Parse error: {e} — skipping")
            continue
        chars = _get_top_speakers(parser, top_n=top_n)
        x_vals, y_vals, y_vars = _extract_scatter_data(
            parser, chars, min_cooc=min_cooc, embedding_type=slug)
        print(f"  {len(x_vals)} pairs (top-{top_n}, min_cooc={min_cooc})")
        feats = _compute_features(x_vals, y_vals, y_vars, y_mean_mode=y_mean_mode)
        if not feats:
            print("  ⚠ Too few pairs — skipping")
            continue
        rec = {"title": title, "genre": genre, "year": year,
               "period": _period_for_year(year), "model": model,
               "y_mean_mode": y_mean_mode}
        rec.update(feats)
        records.append(rec)
        print(f"  y_mean={feats['y_mean']:.4f}  pearson_r={feats['pearson_r']:.4f}")
    _add_isolation_features(records)
    return records, all_interactions


def _add_isolation_features(records):
    """
    Compute per-play isolation features based on corpus-wide distribution.
    'isolation_xy_mean' is the mean Euclidean distance to all other plays
    in the (Year, y_mean) space, with both axes normalized to [0, 1].
    'isolation_dy_mean' is the mean absolute difference in y_mean.
    """
    if not records:
        return
    # Extract years and y_means
    yrs = np.array([r.get("year", np.nan) for r in records], dtype=float)
    yms = np.array([r.get("y_mean", np.nan) for r in records], dtype=float)

    mask = np.isfinite(yrs) & np.isfinite(yms)
    if mask.sum() < 2:
        return

    # Normalized coordinates for distance calculation
    v_yrs = yrs[mask]
    v_yms = yms[mask]

    yr_min, yr_max = v_yrs.min(), v_yrs.max()
    ym_min, ym_max = v_yms.min(), v_yms.max()

    # Min-max scaling
    yr_norm = (v_yrs - yr_min) / (yr_max - yr_min) if yr_max > yr_min else np.zeros_like(v_yrs)
    ym_norm = (v_yms - ym_min) / (ym_max - ym_min) if ym_max > ym_min else np.zeros_like(v_yms)

    # Euclidean distance in normalized (Year, y_mean) space
    xy = np.column_stack([yr_norm, ym_norm])
    diff = xy[:, np.newaxis, :] - xy[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.nan)
    iso_xy = np.nanmean(dist, axis=1)

    # Absolute difference in raw y_mean
    dy = np.abs(v_yms[:, np.newaxis] - v_yms[np.newaxis, :])
    np.fill_diagonal(dy, np.nan)
    iso_dy = np.nanmean(dy, axis=1)

    # Write back to records
    idx = 0
    for i, ok in enumerate(mask):
        if ok:
            records[i]["isolation_xy_mean"] = float(iso_xy[idx])
            records[i]["isolation_dy_mean"] = float(iso_dy[idx])
            idx += 1
        else:
            records[i]["isolation_xy_mean"] = np.nan
            records[i]["isolation_dy_mean"] = np.nan


# ── fitting ──────────────────────────────────────────────────────

def _best_fit(yrs, vals, degree=4):
    """Ridge-regularized polynomial fit. Returns (x_smooth, y_smooth, r2)."""
    if RidgeCV is None or PolynomialFeatures is None:
        return np.array([]), np.array([]), 0.0
    mask = np.isfinite(vals) & np.isfinite(yrs)
    if mask.sum() < 2:
        return np.array([]), np.array([]), 0.0
    yrs, vals = yrs[mask], vals[mask]
    n = len(yrs)
    deg = min(degree, max(1, n - 1))
    yr_min, yr_max = yrs.min(), yrs.max()
    if yr_max <= yr_min:
        return yrs, vals, 0.0
    yr_scaled = (yrs - yr_min) / (yr_max - yr_min)
    poly = PolynomialFeatures(degree=deg, include_bias=True)
    X = poly.fit_transform(yr_scaled.reshape(-1, 1))
    alphas = np.logspace(-4, 2, 50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = RidgeCV(alphas=alphas, cv=min(5, n), fit_intercept=False)
        model.fit(X, vals)
    r2 = float(max(0.0, model.score(X, vals)))
    xs = np.linspace(yr_min, yr_max, 200)
    xs_scaled = (xs - yr_min) / (yr_max - yr_min)
    ys = model.predict(poly.transform(xs_scaled.reshape(-1, 1)))
    return xs, ys, r2


# ── plotting ─────────────────────────────────────────────────────

def _short_title(title):
    """Abbreviate a play title for compact labelling."""
    abbrevs = {
        "The Comedy of Errors": "Err",
        "The Two Gentlemen of Verona": "TGV",
        "The Taming of the Shrew": "Shr",
        "Love's Labour's Lost": "LLL",
        "A Midsummer Night's Dream": "MND",
        "The Merchant of Venice": "MV",
        "The Merry Wives of Windsor": "Wiv",
        "Much Ado About Nothing": "Ado",
        "As You Like It": "AYL",
        "Twelfth Night": "TN",
        "All's Well That Ends Well": "AWW",
        "Measure for Measure": "MM",
        "Titus Andronicus": "Tit",
        "Romeo and Juliet": "Rom",
        "Julius Caesar": "JC",
        "Hamlet": "Ham",
        "Othello": "Oth",
        "King Lear": "Lr",
        "Macbeth": "Mac",
        "Antony and Cleopatra": "Ant",
        "Coriolanus": "Cor",
        "Timon of Athens": "Tim",
        "Troilus and Cressida": "Tro",
        "Henry VI Part 1": "1H6",
        "Henry VI Part 2": "2H6",
        "Henry VI Part 3": "3H6",
        "Richard III": "R3",
        "Richard II": "R2",
        "King John": "Jn",
        "Henry IV Part 1": "1H4",
        "Henry IV Part 2": "2H4",
        "Henry V": "H5",
        "Henry VIII": "H8",
        "Pericles": "Per",
        "Cymbeline": "Cym",
        "The Winter's Tale": "WT",
        "The Tempest": "Tmp",
    }
    return abbrevs.get(title, title[:5])


def _decision_tree_rules_text(clf, feature_names, genre_names):
    """Extract human-readable decision rules from a shallow tree."""
    tree = clf.tree_
    lines = []

    def _recurse(node, depth=0):
        indent = "  " * depth
        if tree.feature[node] != -2:  # not a leaf
            fname = feature_names[tree.feature[node]]
            thresh = tree.threshold[node]
            lines.append(f"{indent}if {fname} <= {thresh:.1f}:")
            _recurse(tree.children_left[node], depth + 1)
            lines.append(f"{indent}else:  # {fname} > {thresh:.1f}")
            _recurse(tree.children_right[node], depth + 1)
        else:
            cls_idx = int(np.argmax(tree.value[node]))
            n_samples = int(tree.n_node_samples[node])
            lines.append(f"{indent}→ {genre_names[cls_idx]} (n={n_samples})")

    _recurse(0)
    return "\n".join(lines)


def plot_chronological_by_genre(records, output_dir, features=None, degree=4):
    if features is None:
        features = DEFAULT_CHRONOLOGICAL_BY_GENRE_FEATURES
    recs = [r for r in records
            if "year" in r and "genre" in r
            and isinstance(r.get("year"), (int, float))
            and np.isfinite(r["year"])]
    if not recs:
        print("  No records with year+genre — nothing to plot.")
        return
    genres = sorted({r["genre"] for r in recs})
    nfeat = len(features)
    colors = {"tragedy": "C3", "comedy": "C2", "history": "C0"}
    fill_colors = {"tragedy": "#d6272822", "comedy": "#2ca02c22", "history": "#1f77b422"}
    markers = {"tragedy": "o", "comedy": "s", "history": "D"}

    if nfeat == 1:
        fig, axes_grid = plt.subplots(2, 1, figsize=(14, 10),
                                      gridspec_kw={"height_ratios": [10, 1]})
        axes = [axes_grid[0]]
        rules_axes = [axes_grid[1]]
    else:
        ncol = 2
        nrow = (nfeat + ncol - 1) // ncol
        fig, axes_arr = plt.subplots(nrow * 2, ncol,
                                     figsize=(8 * ncol, 6.5 * nrow),
                                     gridspec_kw={"height_ratios": [6, 1] * nrow})
        flat = axes_arr.flatten().tolist()
        axes = [flat[j] for j in range(len(flat)) if (j // ncol) % 2 == 0]
        rules_axes = [flat[j] for j in range(len(flat)) if (j // ncol) % 2 == 1]

    all_rules = []

    for i, feat in enumerate(features):
        ax = axes[i]
        rax = rules_axes[i] if i < len(rules_axes) else None

        all_yrs, all_vals, all_genres, all_titles = [], [], [], []
        for r in recs:
            v = r.get(feat, np.nan)
            if np.isfinite(v):
                all_yrs.append(r["year"])
                all_vals.append(v)
                all_genres.append(r["genre"])
                all_titles.append(r["title"])

        text_objs = []
        for g in genres:
            pts = [(yr, v, t) for yr, v, gn, t
                   in zip(all_yrs, all_vals, all_genres, all_titles) if gn == g]
            yrs = np.array([p[0] for p in pts])
            vals = np.array([p[1] for p in pts])
            titles = [p[2] for p in pts]
            ax.scatter(yrs, vals,
                       color=colors.get(g, "gray"),
                       marker=markers.get(g, "o"),
                       alpha=0.7, s=50, label=g, zorder=3)
            for yr, v, t in zip(yrs, vals, titles):
                txt = ax.text(yr, v, t, fontsize=6, alpha=0.85,
                              color=colors.get(g, "gray"), zorder=4)
                text_objs.append(txt)

        if adjust_text is not None and text_objs:
            adjust_text(text_objs, ax=ax,
                        arrowprops=dict(arrowstyle="-", color="gray",
                                        alpha=0.4, lw=0.5))

        # Decision tree boundary
        rules = ""
        if DecisionTreeClassifier is not None and len(all_yrs) >= 5:
            X_dt = np.column_stack([all_yrs, all_vals])
            genre_set = sorted(set(all_genres))
            y_dt = np.array([genre_set.index(g) for g in all_genres])
            dt = DecisionTreeClassifier(max_depth=degree, min_samples_leaf=3)
            dt.fit(X_dt, y_dt)

            yr_margin = (max(all_yrs) - min(all_yrs)) * 0.05
            v_margin = (max(all_vals) - min(all_vals)) * 0.08
            xx, yy = np.meshgrid(
                np.linspace(min(all_yrs) - yr_margin, max(all_yrs) + yr_margin, 300),
                np.linspace(min(all_vals) - v_margin, max(all_vals) + v_margin, 300),
            )
            Z = dt.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            for gi, gname in enumerate(genre_set):
                ax.contourf(xx, yy, (Z == gi).astype(float), levels=[0.5, 1.5],
                            colors=[fill_colors.get(gname, "#88888822")],
                            alpha=0.18, zorder=0)
            ax.contour(xx, yy, Z, colors="gray", linewidths=0.8, alpha=0.5, zorder=1)

            train_acc = dt.score(X_dt, y_dt)
            rules = _decision_tree_rules_text(dt, ["Year", feat], genre_set)
            all_rules.append((feat, rules, train_acc))

        # Rules in a separate panel below the plot
        if rax is not None:
            rax.axis("off")
            if rules:
                acc_pct = train_acc * 100
                header = f"Decision tree (degree {degree}) accuracy for {feat}: {acc_pct:.1f}% on {len(all_yrs)} plays"
                rax.text(0.02, 0.5, header,
                         transform=rax.transAxes,
                         fontsize=10, fontfamily="monospace",
                         verticalalignment="center",
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7",
                                   edgecolor="gray", alpha=0.9))

        ax.set_title(feat, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.25)

    unused_plot = list(range(nfeat, len(axes)))
    unused_rules = list(range(nfeat, len(rules_axes)))
    for j in unused_plot:
        axes[j].set_visible(False)
    for j in unused_rules:
        rules_axes[j].set_visible(False)

    model_label = recs[0].get("model", "bert") if recs else "bert"
    fig.suptitle(f"Chronological trends by genre  (decision tree degree {degree}, {model_label})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    feat_part = "_".join(features) if features else "all"
    from xmlparser import slugify_transformer_model
    model_slug = slugify_transformer_model(model_label)
    basename = f"chronological_by_genre_{feat_part}_{model_slug}.svg"
    path = os.path.join(output_dir, basename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


# ── CLI ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Genre and chronology analysis for Shakespeare play features.")
    ap.add_argument("--data", type=str, default=None,
                    help="Path to data JSON (default: <output-dir>/genre_analysis_data_<model>.json)")
    ap.add_argument("--output-dir", type=str, default="output",
                    help="Directory for data and plots (default: output)")
    ap.add_argument("--plots", type=str, default="chronological_by_genre",
                    help="Comma-separated plot names, or 'all'")
    ap.add_argument("--force-recompute", action="store_true",
                    help="Recompute features from XML even if data file exists")
    ap.add_argument("--features", type=str, default=None,
                    help="Comma-separated features for chronological_by_genre "
                         "(default: y_mean,y_variance_of_averages,y_average_of_variances,y_iqr,x_gini,x_top1_frac,pearson_r,x_mean)")
    ap.add_argument("--degree", type=int, default=4,
                    help="Polynomial degree for trend curves (1–10, default: 4)")
    ap.add_argument("--model", type=str, default="bert",
                    help="Embedding model: registry name (bert, macberth, olmo) "
                         "or any HuggingFace model name (default: bert)")
    ap.add_argument("--y-mean-mode", type=str, choices=["pair", "interaction"], default="pair",
                    help="How to compute y_mean: 'pair' (average of pair averages) "
                         "or 'interaction' (sum of similarities / sum of interactions).")
    ap.add_argument("--separate-plots", action="store_true",
                    help="Save each feature's plot into a separate file.")
    args = ap.parse_args()

    out = args.output_dir.rstrip("/")
    from xmlparser import slugify_transformer_model
    model_slug = slugify_transformer_model(args.model)
    data_path = args.data or os.path.join(out, f"genre_analysis_data_{model_slug}.json")

    if args.force_recompute or not os.path.isfile(data_path):
        records, interactions = compute_all_records(model=args.model, y_mean_mode=args.y_mean_mode)
        os.makedirs(out, exist_ok=True)
        save_records(records, data_path)
        print(f"\nSaved {len(records)} records → {data_path}")

        if interactions:
            import pandas as pd
            csv_path = os.path.join(out, f"speech_interactions_{model_slug}.csv")
            pd.DataFrame(interactions).to_csv(csv_path, index=False)
            print(f"Saved {len(interactions)} speech interactions → {csv_path}")
    else:
        records = load_records(data_path)
        print(f"Loaded {len(records)} records from {data_path}")
        _add_isolation_features(records)

    want = {p.strip() for p in args.plots.split(",")} if args.plots else set()
    if "all" in want:
        want = {"chronological_by_genre"}

    if "chronological_by_genre" in want:
        feats = [x.strip() for x in args.features.split(",")] if args.features else DEFAULT_CHRONOLOGICAL_BY_GENRE_FEATURES
        if args.separate_plots:
            for feat in feats:
                plot_chronological_by_genre(records, out, features=[feat], degree=args.degree)
        else:
            plot_chronological_by_genre(records, out, features=feats, degree=args.degree)


if __name__ == "__main__":
    main()
