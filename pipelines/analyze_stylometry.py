#!/usr/bin/env python3
"""Compute stylometric features and PCA from Folger raw text."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from crelat.catalog import load_play_catalog
from crelat.config import load_config
from crelat.experiment import RunContext
from crelat.features.stylometry import compute_stylometry
from crelat.io.tables import write_table
from crelat.visualization.stylometry import plot_pca

ALLOWED = {"catalog", "raw_text_dir", "play_ids", "n_components"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", default="results/runs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, allowed=ALLOWED, required={"catalog", "raw_text_dir"})
    run = RunContext("analyze-stylometry", config, run_root=args.run_root, force=args.force)
    try:
        selected = set(config.get("play_ids") or [])
        rows = []
        for spec in load_play_catalog(config["catalog"]):
            if selected and spec.id not in selected:
                continue
            source = Path(config["raw_text_dir"]) / f"{spec.folger_code}.txt"
            run.register_input(f"text:{spec.id}", source)
            rows.append(
                {"play_id": spec.id, "title": spec.title, "genre": spec.genre, "year": spec.year, **compute_stylometry(source.read_text(encoding="utf-8", errors="replace"))}
            )
        frame = pd.DataFrame(rows)
        metadata = {"play_id", "title", "genre", "year"}
        columns = [column for column in frame.columns if column not in metadata]
        components = int(config.get("n_components", 2))
        pca = PCA(n_components=components)
        scores = pca.fit_transform(StandardScaler().fit_transform(frame[columns]))
        for index in range(components):
            frame[f"pca{index + 1}"] = scores[:, index]
        long_columns = columns + [f"pca{index + 1}" for index in range(components)]
        long_frame = frame.melt(
            id_vars=["play_id", "title", "genre", "year"],
            value_vars=long_columns,
            var_name="feature",
            value_name="value",
        )
        long_frame["representation"] = "stylometry"
        long_frame["transform"] = "identity"
        long_frame = long_frame[
            ["play_id", "title", "genre", "year", "representation", "feature", "value", "transform"]
        ]
        write_table(
            long_frame,
            run.path / "data" / "stylometry_features.parquet",
            schema="play_features",
        )
        write_table(frame, run.path / "tables" / "stylometry_features.csv")
        pd.DataFrame({"component": [f"pca{i + 1}" for i in range(components)], "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(run.path / "tables" / "explained_variance.csv", index=False)
        plot_pca(frame, run.path / "figures" / "stylometry_pca.svg")
        run.complete({"plays": len(frame)})
        print(run.path)
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
