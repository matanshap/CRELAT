#!/usr/bin/env python3
"""Compare isolation across two canonical representation tables."""

import argparse

from crelat.analysis.comparison import compare_representations
from crelat.config import load_config
from crelat.experiment import RunContext
from crelat.io.tables import read_table, write_table
from crelat.visualization.comparison import plot_isolation_comparison

ALLOWED = {"left_table", "right_table", "left_feature", "right_feature", "left_name", "right_name"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", default="results/runs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, allowed=ALLOWED, required=ALLOWED)
    run = RunContext("compare-representations", config, run_root=args.run_root, force=args.force)
    try:
        left = read_table(config["left_table"])
        right = read_table(config["right_table"])
        run.register_input("left", config["left_table"])
        run.register_input("right", config["right_table"])
        comparison = compare_representations(left, right, left_feature=config["left_feature"], right_feature=config["right_feature"], left_name=config["left_name"], right_name=config["right_name"])
        write_table(comparison, run.path / "data" / "representation_comparison.parquet")
        write_table(comparison, run.path / "tables" / "representation_comparison.csv")
        plot_isolation_comparison(
            comparison,
            run.path / "figures" / "representation_comparison.svg",
            config["left_name"],
            config["right_name"],
        )
        run.complete({"plays": len(comparison)})
        print(run.path)
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
