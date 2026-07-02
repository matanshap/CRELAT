#!/usr/bin/env python3
"""Analyze speech length and TF-IDF against semantic similarity."""

import argparse

from transformers import AutoTokenizer

from crelat.analysis.statistics import cluster_robust_length_regression, length_correlation_results
from crelat.config import load_config
from crelat.experiment import RunContext
from crelat.features.speech_length import (
    add_length_metrics,
    add_tfidf_similarity,
    add_word_count_metrics,
    speech_pair_export,
)
from crelat.io.tables import read_table, write_table
from crelat.visualization.speech_similarity import plot_length_similarity

ALLOWED = {"input_table", "play_id", "tokenizer", "max_sequence_length", "bootstrap_iterations"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", default="results/runs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, allowed=ALLOWED, required={"input_table", "play_id", "tokenizer"})
    run = RunContext("analyze-speech-similarity", config, run_root=args.run_root, force=args.force)
    try:
        frame = read_table(config["input_table"], schema="speech_interactions")
        run.register_input("interactions", config["input_table"])
        frame = frame.loc[frame["play_id"] == config["play_id"]].copy()
        if frame.empty:
            raise ValueError(f"No interactions for play_id {config['play_id']}")
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
        frame = add_tfidf_similarity(frame)
        frame = add_length_metrics(frame, tokenizer, config.get("max_sequence_length"))
        frame = add_word_count_metrics(frame)
        export = speech_pair_export(frame)
        correlations = length_correlation_results(frame, int(config.get("bootstrap_iterations", 5000)))
        regression = cluster_robust_length_regression(frame)
        write_table(frame, run.path / "data" / "speech_similarity.parquet")
        write_table(frame, run.path / "tables" / "speech_similarity.csv")
        write_table(export, run.path / "tables" / "antony_and_cleopatra_speech_pairs.csv")
        write_table(export, run.path / "tables" / "antony_and_cleopatra_speech_pairs.xlsx")
        write_table(correlations, run.path / "tables" / "statistical_results.csv")
        write_table(regression, run.path / "tables" / "regression.csv")
        plot_length_similarity(frame, run.path / "figures" / "speech_length_similarity.svg", config["play_id"])
        run.complete({"interactions": len(frame), "scenes": frame["scene_id"].nunique()})
        print(run.path)
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
