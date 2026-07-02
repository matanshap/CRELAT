#!/usr/bin/env python3
"""Compute play-level semantic features and genre chronology plots."""

import argparse
from collections import Counter

from crelat.analysis.genre import compute_genre_features
from crelat.catalog import load_play_catalog
from crelat.config import load_config
from crelat.experiment import RunContext
from crelat.features.interactions import aggregate_character_pairs
from crelat.io.tables import read_table, write_table
from crelat.visualization.genre import plot_genre_chronology

ALLOWED = {"catalog", "model", "batch_size", "play_ids", "top_n", "min_interactions", "y_mean_mode", "features", "input_table", "speech_table"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", default="results/runs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, allowed=ALLOWED, required={"catalog", "model", "input_table", "speech_table"})
    run = RunContext("analyze-genres", config, run_root=args.run_root, force=args.force)
    try:
        interactions = read_table(config["input_table"], schema="speech_interactions")
        speeches = read_table(config["speech_table"], schema="speeches")
        run.register_input("interactions", config["input_table"])
        run.register_input("speeches", config["speech_table"])
        catalog = load_play_catalog(config["catalog"])
        selected = set(config.get("play_ids") or [])
        if selected:
            catalog = [play for play in catalog if play.id in selected]
        pairs = {}
        for play_id, frame in interactions.groupby("play_id"):
            play_speeches = speeches.loc[
                (speeches["play_id"] == play_id) & (speeches["speaker_id"] != "[UNKNOWN]")
            ]
            counts = Counter(play_speeches["speaker_id"])
            order = play_speeches.groupby("speaker_id")["speaker_order"].min().to_dict()
            ranked = sorted(counts, key=lambda speaker: (-counts[speaker], order[speaker]))
            speakers = ranked[: int(config.get("top_n", 8))]
            pairs[str(play_id)] = aggregate_character_pairs(
                frame,
                speakers=speakers,
                min_interactions=int(config.get("min_interactions", 3)),
            )
        features = compute_genre_features(
            pairs,
            catalog,
            representation=config["model"],
            y_mean_mode=config.get("y_mean_mode", "pair"),
        )
        write_table(features, run.path / "data" / "play_features.parquet", schema="play_features")
        write_table(features, run.path / "tables" / "play_features.csv")
        for feature in config.get("features", ["y_mean"]):
            plot_genre_chronology(features, feature, run.path / "figures" / f"{feature}.svg")
        run.complete({"plays": features["play_id"].nunique()})
        print(run.path)
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
