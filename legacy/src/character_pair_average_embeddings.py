#!/usr/bin/env python3
"""
Average speech-embedding midpoint for each character pair among the top-N speakers.

For every consecutive speech boundary where the two speakers are exactly the pair
(A, B) (in either order), we take (e_i + e_{i+1}) / 2, then average those vectors
over all such boundaries. Uses HuggingFace transformer mean-pooled embeddings via
XMLParser(transformer_model_name=...).

Example (from repo root, with PYTHONPATH=src or run inside src):

  cd src && python character_pair_average_embeddings.py \\
    ../Data/XML/folger_corpus/Ham.xml --model bert-base-uncased --top-n 8 \\
    -o ../output/Hamlet_pair_embeddings_bert.json

By default also writes an interactions scatter SVG matching
XMLParser.plot_bert_interactions_scatter_normalized: X = co-occurrence count,
Y = (sum of consecutive-speech cosine similarities) / interactions.
Two isolation scatters use the same pairs: mean distance in the (interactions, Y) plane to
other pairs, and mean absolute Delta-Y vs. other pairs (written next to the main SVG).
Use --no-plot to skip, or --plot PATH for a custom path to the main scatter only.

Repeat ``--model`` for several models on one play (JSON/plots use default names under
``output/``; do not use ``-o`` or ``--plot`` with multiple models). For a full corpus
and a combined PPTX, use ``corpus_multi_model_interactions.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np
import torch.nn.functional as F

from xmlparser import XMLParser, read_tei_play_title


def _vec_to_numpy(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float64)


def top_speakers_by_speech_count(parser, top_n: int, include_unknown: bool = False):
    speech_counts = {char: 0 for char in parser.characters}
    for scene in parser.characters_speeches:
        for speech in scene:
            speaker = speech.get("speaker", "[UNKNOWN]")
            if speaker in speech_counts:
                speech_counts[speaker] += 1
    ranked = sorted(speech_counts.items(), key=lambda item: item[1], reverse=True)
    if not include_unknown:
        ranked = [item for item in ranked if item[0] != "[UNKNOWN]"]
    return [char for char, _ in ranked[:top_n]]


def interaction_cosine_sum_matrix(parser, embedding_key: str):
    """
    Same accumulation as XMLParser.__calculate_cosine_similarity: sum of consecutive-speech
    cosine similarities per unordered pair (symmetric matrix).
    """
    chars = parser.characters
    cosine_similarities = {c: {c: 0.0 for c in chars} for c in chars}
    for scene in parser.characters_speeches:
        for speech_idx in range(len(scene) - 1):
            speaker = scene[speech_idx]["speaker"]
            next_speaker = scene[speech_idx + 1]["speaker"]
            if speaker == next_speaker:
                continue
            e1 = scene[speech_idx].get(embedding_key)
            e2 = scene[speech_idx + 1].get(embedding_key)
            if e1 is None or e2 is None:
                continue
            cosine_similarities[speaker][next_speaker] += F.cosine_similarity(
                e1.unsqueeze(0), e2.unsqueeze(0)
            ).item()
            cosine_similarities[next_speaker][speaker] = cosine_similarities[speaker][next_speaker]
    return cosine_similarities


def pair_average_midpoint_embeddings(parser, embedding_key: str, top_characters: list):
    """
    Unordered pairs from top_characters. Mean of (e1+e2)/2 over consecutive
    cross-speaker turns where the speaker set equals the pair.
    """
    chars = [c for c in top_characters if c in parser.characters and c != "[UNKNOWN]"]
    results = []
    for a, b in combinations(sorted(chars), 2):
        mids = []
        for scene in parser.characters_speeches:
            for i in range(len(scene) - 1):
                s1, s2 = scene[i], scene[i + 1]
                sp1, sp2 = s1.get("speaker"), s2.get("speaker")
                if sp1 == sp2:
                    continue
                if {sp1, sp2} != {a, b}:
                    continue
                e1 = s1.get(embedding_key)
                e2 = s2.get(embedding_key)
                if e1 is None or e2 is None:
                    continue
                v1, v2 = _vec_to_numpy(e1), _vec_to_numpy(e2)
                mids.append((v1 + v2) / 2.0)
        if not mids:
            mean_vec = None
            count = 0
        else:
            mean_vec = np.mean(np.stack(mids, axis=0), axis=0)
            count = len(mids)
        results.append(
            {
                "character_a": a,
                "character_b": b,
                "interaction_count": count,
                "mean_midpoint_embedding": mean_vec.tolist() if mean_vec is not None else None,
            }
        )
    return results


def run_single_play_model(
    xml_path: str,
    model_name: str,
    *,
    output_dir: str = "output",
    write_json: bool = True,
    write_plot: bool = True,
    json_path: str | None = None,
    plot_path: str | None = None,
    top_n: int = 8,
    batch_size: int = 16,
    include_unknown: bool = False,
    play_title: str | None = None,
    min_cooc: int = 0,
) -> dict:
    """
    Parse one play, embed speeches with ``model_name``, optional JSON + normalized interactions plot.

    Returns metadata including paths written (or None if skipped).
    """
    xml_path = os.path.abspath(xml_path)
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    parser = XMLParser(
        xml_path,
        options={"co-oc"},
        transformer_model_name=model_name,
        embedding_batch_size=batch_size,
    )
    parser.parse()

    if not parser._transformer_embedding_slug:
        raise RuntimeError("Transformer embeddings were not attached to speeches.")

    embedding_key = f"average_{parser._transformer_embedding_slug}_embedding"
    top_chars = top_speakers_by_speech_count(
        parser, top_n=top_n, include_unknown=include_unknown
    )
    pairs = pair_average_midpoint_embeddings(parser, embedding_key, top_chars)

    play_stem = os.path.splitext(os.path.basename(xml_path))[0]
    if (play_title or "").strip():
        title = play_title.strip()
    else:
        tei_title = read_tei_play_title(xml_path)
        title = tei_title if tei_title else play_stem
    slug = parser._transformer_embedding_slug

    payload = {
        "xml_path": xml_path,
        "play_code": play_stem,
        "model_name": model_name,
        "embedding_key": embedding_key,
        "top_n": top_n,
        "top_characters": top_chars,
        "pairs": pairs,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_json = None
    if write_json:
        out_json = os.path.abspath(
            json_path
            if json_path
            else os.path.join(output_dir, f"{play_stem}_{slug}_pairs.json")
        )
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    out_plot = None
    out_plot_isolation_xy = None
    out_plot_isolation_dy = None
    if write_plot:
        out_plot = os.path.abspath(
            plot_path
            if plot_path
            else os.path.join(
                output_dir, f"{play_stem}_{slug}_interactions_scatter_normalized.svg"
            )
        )
        os.makedirs(os.path.dirname(out_plot) or ".", exist_ok=True)
        cos_matrix = interaction_cosine_sum_matrix(parser, embedding_key)
        y_label = f"Cosine similarity / interactions ({model_name})"
        filename_suffix = f"{slug}_interactions_scatter_normalized"
        parser._plot_interactions_scatter(
            cos_matrix,
            title,
            y_label=y_label,
            filename_suffix=filename_suffix,
            output_dir=output_dir,
            characters_filter=top_chars,
            use_y_ratio=True,
            use_softmax=False,
            show_trend=False,
            min_cooc_threshold=min_cooc,
            output_path=out_plot,
        )
        if not os.path.isfile(out_plot):
            out_plot = None

        out_plot_isolation_xy = os.path.join(
            output_dir, f"{play_stem}_{slug}_interactions_isolation_xy.svg"
        )
        out_plot_isolation_dy = os.path.join(
            output_dir, f"{play_stem}_{slug}_interactions_isolation_dy.svg"
        )
        parser._plot_interactions_isolation_scatters(
            cos_matrix,
            title,
            ref_y_label=y_label,
            characters_filter=top_chars,
            min_cooc_threshold=min_cooc,
            use_y_ratio=True,
            output_path_xy=out_plot_isolation_xy,
            output_path_dy=out_plot_isolation_dy,
        )
        out_plot_isolation_xy = (
            os.path.abspath(out_plot_isolation_xy)
            if os.path.isfile(out_plot_isolation_xy)
            else None
        )
        out_plot_isolation_dy = (
            os.path.abspath(out_plot_isolation_dy)
            if os.path.isfile(out_plot_isolation_dy)
            else None
        )

    return {
        "xml_path": xml_path,
        "play_stem": play_stem,
        "play_title": title,
        "model_name": model_name,
        "slug": slug,
        "embedding_key": embedding_key,
        "json_path": out_json,
        "plot_path": out_plot,
        "plot_path_isolation_xy": out_plot_isolation_xy,
        "plot_path_isolation_dy": out_plot_isolation_dy,
        "top_characters": top_chars,
        "payload": payload,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Mean interaction embedding per character pair (top-N speakers, HuggingFace model)."
    )
    ap.add_argument("xml_path", help="Path to Folger TEI XML (e.g. Data/XML/.../Ham.xml)")
    ap.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="MODEL",
        help="HuggingFace model id (repeat for several models, e.g. --model bert-base-uncased --model gpt2)",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of most frequent speakers (by speech count) to include (default: 8)",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="",
        help="Write JSON to this path (default: print to stdout)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding inference (default: 16)",
    )
    ap.add_argument(
        "--include-unknown",
        action="store_true",
        help="Allow [UNKNOWN] in top-N ranking",
    )
    ap.add_argument(
        "--plot",
        default="",
        metavar="PATH",
        help="Save interactions scatter (SVG), same style as plot_*_interactions_scatter_normalized. "
        "Default: output/<play>_<model_slug>_interactions_scatter_normalized.svg",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write a figure",
    )
    ap.add_argument(
        "--play-name",
        default="",
        help="Title/filename prefix for the plot (default: XML basename without .xml)",
    )
    ap.add_argument(
        "--min-cooc",
        type=int,
        default=0,
        metavar="N",
        help="Minimum co-occurrence count to include a pair in the plot (default: 0)",
    )
    args = ap.parse_args()

    models = args.models
    if not models:
        raise SystemExit("Provide at least one --model (e.g. --model bert-base-uncased).")

    xml_path = os.path.abspath(args.xml_path)
    if not os.path.isfile(xml_path):
        raise SystemExit(f"XML file not found: {xml_path}")

    play_title = args.play_name.strip() or None

    if args.output and len(models) > 1:
        raise SystemExit("With multiple --model, omit -o (JSON files use default names under output/).")
    if args.plot and len(models) > 1:
        raise SystemExit("With multiple --model, omit --plot (plots use default names under output/).")

    for model_name in models:
        custom_json = os.path.abspath(args.output) if args.output else None
        custom_plot = os.path.abspath(args.plot) if args.plot else None
        write_json = bool(args.output) or len(models) > 1

        info = run_single_play_model(
            xml_path,
            model_name,
            output_dir="output",
            write_json=write_json,
            write_plot=not args.no_plot,
            json_path=custom_json,
            plot_path=custom_plot,
            top_n=args.top_n,
            batch_size=args.batch_size,
            include_unknown=args.include_unknown,
            play_title=play_title,
            min_cooc=args.min_cooc,
        )

        if write_json:
            print(f"Wrote {info['json_path']}")
        elif len(models) == 1:
            print(json.dumps(info["payload"], indent=2))

        if info["plot_path"]:
            print(f"Wrote plot {info['plot_path']}")
        if info.get("plot_path_isolation_xy"):
            print(f"Wrote plot {info['plot_path_isolation_xy']}")
        if info.get("plot_path_isolation_dy"):
            print(f"Wrote plot {info['plot_path_isolation_dy']}")


if __name__ == "__main__":
    main()
