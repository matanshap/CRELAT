#!/usr/bin/env python3
"""Build canonical speech-interaction tables from Folger plays."""

import argparse
from pathlib import Path

import pandas as pd

from crelat.catalog import load_play_catalog
from crelat.config import load_config
from crelat.embeddings import create_embedder
from crelat.experiment import RunContext
from crelat.features.interactions import build_speech_interactions
from crelat.io.folger import parse_play
from crelat.io.tables import write_table

ALLOWED = {"catalog", "model", "batch_size", "play_ids", "top_n", "min_interactions", "y_mean_mode", "features", "input_table", "speech_table"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", default="results/runs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, allowed=ALLOWED, required={"catalog", "model"})
    run = RunContext("build-interactions", config, run_root=args.run_root, force=args.force)
    try:
        catalog = load_play_catalog(config["catalog"], require_files=True)
        selected = set(config.get("play_ids") or [])
        plays = [spec for spec in catalog if not selected or spec.id in selected]
        embedder = create_embedder(config["model"], batch_size=int(config.get("batch_size", 16)))
        frames = []
        speech_frames = []
        for spec in plays:
            run.register_input(f"xml:{spec.id}", spec.xml)
            play = parse_play(spec)
            speaker_order = {speaker: index for index, speaker in enumerate(play.characters)}
            speech_frames.append(
                pd.DataFrame(
                    [
                        {
                            "speech_id": speech.id,
                            "play_id": speech.play_id,
                            "scene_id": speech.scene_id,
                            "position": speech.position,
                            "speaker_id": speech.speaker_id,
                            "speaker_order": speaker_order[speech.speaker_id],
                            "text": speech.text,
                        }
                        for speech in play.speeches
                    ]
                )
            )
            embeddings = embedder.embed([speech.text for speech in play.speeches])
            frames.append(build_speech_interactions(play, embeddings, config["model"]))
        interactions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        speeches = pd.concat(speech_frames, ignore_index=True) if speech_frames else pd.DataFrame()
        write_table(
            interactions,
            run.path / "data" / "speech_interactions.parquet",
            schema="speech_interactions",
        )
        write_table(interactions, run.path / "tables" / "speech_interactions.csv")
        write_table(speeches, run.path / "data" / "speeches.parquet", schema="speeches")
        run.complete({"plays": len(plays), "speeches": len(speeches), "interactions": len(interactions)})
        print(run.path)
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
