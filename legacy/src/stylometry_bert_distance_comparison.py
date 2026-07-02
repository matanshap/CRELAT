#!/usr/bin/env python3
"""
Compare play isolation in the stylometry PCA plot and the BERT y_mean plot.

For each play, compute its mean absolute vertical distance from all other plays
in each plot. The output scatter uses:
  X = mean y-distance in the stylometry PCA1 chronology plot
  Y = mean y-distance in the BERT y_mean chronology plot
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

try:
    from genre_analysis import _short_title
except ImportError:
    def _short_title(title):
        return title[:5]


def _read_stylometry_pca(path):
    records = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                pca1 = float(row["pca1"])
            except (KeyError, TypeError, ValueError):
                continue
            title = row.get("title", "")
            if not title:
                continue
            records[title] = {
                "title": title,
                "genre": row.get("genre", ""),
                "year": int(float(row["year"])) if row.get("year") else "",
                # Match src/stylometry_pca.py, which plots all_vals = -pca1.
                "stylometry_y": -pca1,
            }
    return records


def _read_bert_records(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = {}
    for row in raw:
        title = row.get("title", "")
        if not title:
            continue
        try:
            y_mean = float(row["y_mean"])
        except (KeyError, TypeError, ValueError):
            continue
        records[title] = {
            "title": title,
            "genre": row.get("genre", ""),
            "year": int(float(row["year"])) if row.get("year") else "",
            "bert_y_mean": y_mean,
        }
    return records


def _mean_abs_distances(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.full(len(values), np.nan)
    diff = np.abs(values[:, np.newaxis] - values[np.newaxis, :])
    np.fill_diagonal(diff, np.nan)
    return np.nanmean(diff, axis=1)


def build_comparison_rows(stylometry_records, bert_records):
    titles = sorted(set(stylometry_records) & set(bert_records))
    rows = []
    for title in titles:
        s = stylometry_records[title]
        b = bert_records[title]
        rows.append({
            "title": title,
            "genre": s.get("genre") or b.get("genre", ""),
            "year": s.get("year") or b.get("year", ""),
            "stylometry_y": float(s["stylometry_y"]),
            "bert_y_mean": float(b["bert_y_mean"]),
        })

    stylometry_dist = _mean_abs_distances([r["stylometry_y"] for r in rows])
    bert_dist = _mean_abs_distances([r["bert_y_mean"] for r in rows])
    for row, s_dist, b_dist in zip(rows, stylometry_dist, bert_dist):
        row["avg_stylometry_y_distance"] = float(s_dist)
        row["avg_bert_y_mean_distance"] = float(b_dist)

    rows.sort(key=lambda r: (r["avg_stylometry_y_distance"], r["title"]))
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "title", "genre", "year", "stylometry_y", "bert_y_mean",
        "avg_stylometry_y_distance", "avg_bert_y_mean_distance",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_comparison_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "title": row["title"],
                    "genre": row["genre"],
                    "year": int(float(row["year"])) if row.get("year") else "",
                    "stylometry_y": float(row["stylometry_y"]),
                    "bert_y_mean": float(row["bert_y_mean"]),
                    "avg_stylometry_y_distance": float(row["avg_stylometry_y_distance"]),
                    "avg_bert_y_mean_distance": float(row["avg_bert_y_mean_distance"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def plot_comparison(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    colors = {"tragedy": "C3", "comedy": "C2", "history": "C0"}
    markers = {"tragedy": "o", "comedy": "s", "history": "D"}
    genres = sorted({r["genre"] for r in rows})

    fig, (ax, glossary_ax) = plt.subplots(
        1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [4.2, 1.8]}
    )
    text_objs = []
    for genre in genres:
        pts = [r for r in rows if r["genre"] == genre]
        xs = [r["avg_stylometry_y_distance"] for r in pts]
        ys = [r["avg_bert_y_mean_distance"] for r in pts]
        ax.scatter(
            xs, ys, s=70, alpha=0.75, label=genre,
            color=colors.get(genre, "gray"),
            marker=markers.get(genre, "o"),
            edgecolors="white", linewidths=0.8,
        )
        for x, y, row in zip(xs, ys, pts):
            text_objs.append(
                ax.text(
                    x, y, _short_title(row["title"]), fontsize=8,
                    color=colors.get(genre, "gray"), alpha=0.9,
                )
            )

    if adjust_text is not None and text_objs:
        adjust_text(
            text_objs, ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.35, lw=0.5),
        )

    ax.set_title(
        "Play Isolation: Stylometry PCA vs BERT y_mean",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Average stylometry y-distance from other plays (-PCA1)")
    ax.set_ylabel("Average BERT y_mean distance from other plays")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")

    glossary_ax.axis("off")
    glossary_ax.set_title("Point Label Glossary", fontsize=11, fontweight="bold", loc="left")
    glossary = sorted((_short_title(r["title"]), r["title"]) for r in rows)
    midpoint = (len(glossary) + 1) // 2
    columns = [glossary[:midpoint], glossary[midpoint:]]
    for col_idx, entries in enumerate(columns):
        x = 0.02 + col_idx * 0.49
        y = 0.96
        for short, title in entries:
            glossary_ax.text(
                x, y, f"{short}: {title}",
                transform=glossary_ax.transAxes,
                fontsize=7.2,
                va="top",
            )
            y -= 0.048

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare average y-distances in stylometry PCA and BERT y_mean plots."
    )
    parser.add_argument(
        "--stylometry-pca",
        default="output/stylometry_pca_results.csv",
        help="CSV from src/stylometry_pca.py.",
    )
    parser.add_argument(
        "--bert-data",
        default="output/genre_analysis_data_bert.json",
        help="BERT genre-analysis JSON with y_mean.",
    )
    parser.add_argument(
        "--output-csv",
        default="output/stylometry_vs_bert_y_distance.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-plot",
        default="output/stylometry_vs_bert_y_distance.svg",
        help="Output SVG path.",
    )
    args = parser.parse_args()

    if os.path.exists(args.stylometry_pca) and os.path.exists(args.bert_data):
        stylometry_records = _read_stylometry_pca(args.stylometry_pca)
        bert_records = _read_bert_records(args.bert_data)
        rows = build_comparison_rows(stylometry_records, bert_records)
        write_csv(rows, args.output_csv)
    elif os.path.exists(args.output_csv):
        rows = read_comparison_csv(args.output_csv)
        print(f"Raw input missing; redrawing from existing CSV {args.output_csv}")
    else:
        raise SystemExit(
            "Missing raw inputs and no existing comparison CSV to redraw from."
        )
    if len(rows) < 2:
        raise SystemExit("Need at least two plays present in both input files.")

    plot_comparison(rows, args.output_plot)
    print(f"Compared {len(rows)} plays")
    print(f"Wrote CSV to {args.output_csv}")
    print(f"Wrote plot to {args.output_plot}")


if __name__ == "__main__":
    main()
