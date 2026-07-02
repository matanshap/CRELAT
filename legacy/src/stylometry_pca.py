#!/usr/bin/env python3
"""
PCA analysis of Shakespeare stylometric features.
Reduces stylometric dimensions to PCA1/PCA2 and plots the components and chronology.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
import warnings

# Use Agg backend for headless environments
import matplotlib
matplotlib.use("Agg")

try:
    from sklearn.tree import DecisionTreeClassifier
except ImportError:
    DecisionTreeClassifier = None

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

def _decision_tree_rules_text(clf, feature_names, genre_names):
    """Extract human-readable decision rules from a shallow tree."""
    tree = clf.tree_
    lines = []

    def _recurse(node, depth=0):
        indent = "  " * depth
        if tree.feature[node] != -2:  # not a leaf
            fname = feature_names[tree.feature[node]]
            thresh = tree.threshold[node]
            lines.append(f"{indent}if {fname} <= {thresh:.3f}:")
            _recurse(tree.children_left[node], depth + 1)
            lines.append(f"{indent}else:  # {fname} > {thresh:.3f}")
            _recurse(tree.children_right[node], depth + 1)
        else:
            cls_idx = int(np.argmax(tree.value[node]))
            n_samples = int(tree.n_node_samples[node])
            lines.append(f"{indent}→ {genre_names[cls_idx]} (n={n_samples})")

    _recurse(0)
    return "\n".join(lines)


def plot_pca_components(records, output_path, explained_variance=None):
    """Plot PCA1 against PCA2, grouped by genre and labeled by play."""
    valid = [
        r for r in records
        if "genre" in r
        and np.isfinite(r.get("pca1", np.nan))
        and np.isfinite(r.get("pca2", np.nan))
    ]
    if not valid:
        print("No valid records for PCA1 vs. PCA2 plotting.")
        return

    colors = {"tragedy": "C3", "comedy": "C2", "history": "C0"}
    markers = {"tragedy": "o", "comedy": "s", "history": "D"}

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 9))
    text_objs = []

    for genre in sorted({r["genre"] for r in valid}):
        genre_records = [r for r in valid if r["genre"] == genre]
        pca1 = np.array([r["pca1"] for r in genre_records])
        pca2 = np.array([r["pca2"] for r in genre_records])
        ax.scatter(
            pca1,
            pca2,
            color=colors.get(genre, "gray"),
            marker=markers.get(genre, "o"),
            alpha=0.75,
            s=60,
            label=genre,
            zorder=3,
        )
        for x, y, record in zip(pca1, pca2, genre_records):
            text_objs.append(
                ax.text(
                    x,
                    y,
                    record.get("title", ""),
                    fontsize=7,
                    alpha=0.85,
                    color=colors.get(genre, "gray"),
                    zorder=4,
                )
            )

    if adjust_text is not None and text_objs:
        adjust_text(
            text_objs,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.4, lw=0.5),
        )

    axis_labels = ["PCA1", "PCA2"]
    if explained_variance is not None and len(explained_variance) >= 2:
        axis_labels = [
            f"PCA1 ({explained_variance[0] * 100:.1f}% variance)",
            f"PCA2 ({explained_variance[1] * 100:.1f}% variance)",
        ]

    ax.axhline(0, color="gray", linewidth=0.7, alpha=0.35, zorder=1)
    ax.axvline(0, color="gray", linewidth=0.7, alpha=0.35, zorder=1)
    ax.set_title(
        "Shakespeare Stylometry: PCA Component 1 vs. Component 2",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_pca_chronology(records, output_path, loadings=None, degree=4):
    """Plots PCA1 vs Year matching the genre_analysis style."""
    
    # Filter records with year and genre
    valid = [r for r in records if "year" in r and "genre" in r and np.isfinite(r["pca1"])]
    if not valid:
        print("No valid records for plotting.")
        return

    all_yrs = np.array([r["year"] for r in valid])
    all_vals = -np.array([r["pca1"] for r in valid])
    all_genres = [r["genre"] for r in valid]
    all_titles = [r["title"] for r in valid]
    
    genres = sorted(set(all_genres))
    
    # Plotting setup - matching genre_analysis.py
    plt.style.use('default')
    fig, axes_grid = plt.subplots(2, 1, figsize=(14, 10),
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax = axes_grid[0]
    rax = axes_grid[1]
    
    colors = {"tragedy": "C3", "comedy": "C2", "history": "C0"}
    fill_colors = {"tragedy": "#d6272822", "comedy": "#2ca02c22", "history": "#1f77b422"}
    markers = {"tragedy": "o", "comedy": "s", "history": "D"}
    
    # Scatter plot
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
            txt = ax.text(yr, v, t, fontsize=7, alpha=0.85,
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
        dt = DecisionTreeClassifier(max_depth=degree, min_samples_leaf=2)
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
        rules = _decision_tree_rules_text(dt, ["Year", "PCA1"], genre_set)
        
        # Rules panel
        rax.axis("off")
        if rules:
            acc_pct = train_acc * 100
            header = f"Decision tree (degree {degree}) accuracy for PCA1: {acc_pct:.1f}% on {len(all_yrs)} plays"
            rax.text(0.02, 0.95, header,
                     transform=rax.transAxes,
                     fontsize=10, fontfamily="monospace",
                     verticalalignment="top",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7",
                               edgecolor="gray", alpha=0.9))
            rax.text(0.02, 0.70, rules,
                     transform=rax.transAxes,
                     fontsize=9, fontfamily="monospace",
                     verticalalignment="top")

    # Add PCA loadings text
    if loadings:
        ev_ratio = loadings.get("__explained_variance__", 0.0)
        loading_text = f"PCA1: {ev_ratio*100:.1f}% Variance\n"
        loading_text += "Top Loadings:\n"
        # Sort by absolute value, excluding the special key
        sorted_loadings = sorted([(k, -v) for k, v in loadings.items() if k != "__explained_variance__"],
                                  key=lambda x: abs(x[1]), reverse=True)
        for feat, val in sorted_loadings[:8]:
            loading_text += f"  {feat}: {val:+.3f}\n"
        
        rax.text(0.60, 0.95, loading_text,
                 transform=rax.transAxes,
                 fontsize=9, fontfamily="monospace",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f8ff",
                           edgecolor="blue", alpha=0.8))

    # Labels and title
    ax.set_title("Shakespeare Stylometry: PCA Component 1 vs. Year", fontsize=14, fontweight='bold')
    ax.set_xlabel("Year")
    ax.set_ylabel("PCA1")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PCA analysis of Shakespeare stylometry")
    parser.add_argument("--degree", type=int, default=4, help="Decision tree max depth (degree)")
    parser.add_argument("--data", type=str, default="output/stylometry_analysis_data.json", help="Input data path")
    parser.add_argument("--output-plot", type=str, default="output/stylometry_pca1_vs_year.svg", help="Output plot path")
    parser.add_argument("--output-pca-plot", type=str, default="output/stylometry_pca1_vs_pca2.svg", help="PCA1 vs. PCA2 output plot path")
    parser.add_argument("--output-csv", type=str, default="output/stylometry_pca_results.csv", help="Output CSV path")
    args = parser.parse_args()

    data_path = args.data
    output_plot = args.output_plot
    output_pca_plot = args.output_pca_plot
    output_csv = args.output_csv
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run stylometry_analysis.py first.")
        return

    with open(data_path, "r") as f:
        records = json.load(f)

    features_to_pca = [
        "bws_types_per_1k", "mean_letters", "personal_pronouns", 
        "pronoun_1st_sing", "pronoun_1st_plur", "pronoun_2nd_sing", "pronoun_2nd_plur", "pronoun_3rd",
        "determiners", "prepositions", "conjunctions", "aux_verbs", "negations",
        "positive_emotions", "negative_emotions", "certainty_words", "cognitive_words", "social_words", "family_words"
    ]

    # Build feature matrix
    X = []
    valid_records = []
    for r in records:
        try:
            vec = [float(r[f]) for f in features_to_pca]
            X.append(vec)
            valid_records.append(r)
        except (KeyError, TypeError):
            continue

    X = np.array(X)
    if X.size == 0:
        print("No features found to perform PCA.")
        return

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA Explained Variance: {pca.explained_variance_ratio_}")
    print(f"PCA1: {pca.explained_variance_ratio_[0]*100:.1f}%, PCA2: {pca.explained_variance_ratio_[1]*100:.1f}%")

    # Get PCA1 loadings
    loadings = dict(zip(features_to_pca, pca.components_[0]))
    loadings["__explained_variance__"] = pca.explained_variance_ratio_[0]

    # Add PCA results back to records
    for i, r in enumerate(valid_records):
        r["pca1"] = X_pca[i, 0]
        r["pca2"] = X_pca[i, 1]

    # Export to CSV
    try:
        df_export = pd.DataFrame(valid_records)
        df_export.to_csv(output_csv, index=False)
        print(f"Saved PCA results and features to {output_csv}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    # Plot
    plot_pca_chronology(valid_records, output_plot, loadings=loadings, degree=args.degree)
    plot_pca_components(
        valid_records,
        output_pca_plot,
        explained_variance=pca.explained_variance_ratio_,
    )


if __name__ == "__main__":
    main()
