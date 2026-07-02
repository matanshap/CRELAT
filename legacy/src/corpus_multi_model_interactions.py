#!/usr/bin/env python3
"""
For each TEI XML in a corpus and each HuggingFace embedding model, run the same pipeline as
character_pair_average_embeddings.py (pair JSON + normalized interactions scatter SVG).

Optionally build a PowerPoint with one slide per play; each slide shows all models' plots
side by side (requires python-pptx and cairosvg).

Typical usage (from ``src/``):

  python corpus_multi_model_interactions.py \\
    --corpus-dir ../Data/XML/folger_corpus \\
    --models bert-base-uncased gpt2 \\
    --output-dir ../output \\
    --ppt ../output/corpus_interactions_models.pptx
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from character_pair_average_embeddings import run_single_play_model
from xmlparser import read_tei_play_title, slugify_transformer_model


def load_play_titles_map(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Play-names JSON must be an object: {\"Ham\": \"Hamlet\", ...}")
    return {str(k): str(v) for k, v in data.items()}


def collect_xml_paths(
    *,
    corpus_dir: str | None = None,
    corpus_list: str | None = None,
    xml_paths: list[str] | None = None,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add(path: str) -> None:
        ap = os.path.abspath(os.path.expanduser(path.strip()))
        if ap in seen:
            return
        if os.path.isfile(ap):
            seen.add(ap)
            out.append(ap)
        else:
            print(f"Warning: skip missing XML: {path}")

    if corpus_list:
        with open(corpus_list, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                add(line)

    if corpus_dir:
        pattern = os.path.join(os.path.expanduser(corpus_dir), "*.xml")
        for p in sorted(glob.glob(pattern)):
            add(p)

    if xml_paths:
        for p in xml_paths:
            add(p)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch pair embeddings + interaction plots over a corpus and multiple models."
    )
    ap.add_argument(
        "--corpus-dir",
        metavar="DIR",
        help="Directory containing *.xml (Folger corpus folder)",
    )
    ap.add_argument(
        "--corpus-list",
        metavar="FILE",
        help="Text file: one XML path per line (# comments allowed)",
    )
    ap.add_argument(
        "--xml",
        action="append",
        dest="xml_extra",
        default=[],
        metavar="PATH",
        help="Extra XML path (repeatable)",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="MODEL",
        help="HuggingFace model ids, same order as columns on each PPT slide",
    )
    ap.add_argument(
        "--output-dir",
        default="output",
        help="Directory for JSON and SVG outputs (default: output)",
    )
    ap.add_argument(
        "--ppt",
        metavar="PATH.pptx",
        help="Write a presentation: one slide per play, all models on that slide",
    )
    ap.add_argument(
        "--manifest",
        metavar="PATH.json",
        help="Write a JSON manifest of slides, models, and artifact paths",
    )
    ap.add_argument(
        "--play-names-json",
        metavar="FILE",
        help='Optional JSON mapping XML basename to display title (overrides TEI <titleStmt><title>)',
    )
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--min-cooc", type=int, default=0)
    ap.add_argument("--no-json", action="store_true", help="Skip writing pair JSON files")
    ap.add_argument("--no-plot", action="store_true", help="Skip SVG plots (incompatible with --ppt)")
    ap.add_argument("--raster-dpi", type=int, default=192, help="SVG rasterization DPI for PPT")
    args = ap.parse_args()

    if args.no_plot and args.ppt:
        raise SystemExit("--ppt requires plots; omit --no-plot.")

    xml_paths = collect_xml_paths(
        corpus_dir=args.corpus_dir,
        corpus_list=args.corpus_list,
        xml_paths=args.xml_extra or None,
    )
    if not xml_paths:
        raise SystemExit(
            "No XML files found. Use --corpus-dir, --corpus-list, and/or --xml PATH."
        )

    model_slugs = [slugify_transformer_model(m) for m in args.models]
    if len(model_slugs) != len(set(model_slugs)):
        raise SystemExit(
            "--models must map to unique filename slugs (no two models with the same slug)."
        )

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    title_map: dict[str, str] = {}
    if args.play_names_json:
        title_map = load_play_titles_map(args.play_names_json)

    play_meta: dict[str, dict] = {}
    for xml_path in xml_paths:
        play_stem = os.path.splitext(os.path.basename(xml_path))[0]
        slide_title = title_map.get(play_stem) or read_tei_play_title(xml_path) or play_stem
        play_meta[xml_path] = {"play_stem": play_stem, "slide_title": slide_title}

    # model outer loop: each HF model stays loaded while we sweep the corpus (cached singleton).
    by_play_model: dict[tuple[str, str], dict] = {}

    for model_name in args.models:
        for xml_path in xml_paths:
            meta = play_meta[xml_path]
            info = run_single_play_model(
                xml_path,
                model_name,
                output_dir=output_dir,
                write_json=not args.no_json,
                write_plot=not args.no_plot,
                top_n=args.top_n,
                batch_size=args.batch_size,
                play_title=meta["slide_title"],
                min_cooc=args.min_cooc,
            )
            by_play_model[(xml_path, model_name)] = info
            if info["json_path"]:
                print(f"Wrote {info['json_path']}")
            if info["plot_path"]:
                print(f"Wrote plot {info['plot_path']}")
            if info.get("plot_path_isolation_xy"):
                print(f"Wrote plot {info['plot_path_isolation_xy']}")
            if info.get("plot_path_isolation_dy"):
                print(f"Wrote plot {info['plot_path_isolation_dy']}")

    slides_spec: list[dict] = []
    manifest: dict = {
        "models": list(args.models),
        "output_dir": output_dir,
        "slides": [],
    }

    for xml_path in xml_paths:
        meta = play_meta[xml_path]
        slide_title = meta["slide_title"]
        panels: list[dict] = []
        slide_entry: dict = {
            "xml_path": xml_path,
            "play_stem": meta["play_stem"],
            "title": slide_title,
            "models": [],
        }
        for model_name in args.models:
            info = by_play_model[(xml_path, model_name)]
            if info["plot_path"]:
                panels.append({"label": model_name, "svg": info["plot_path"]})
            if info.get("plot_path_isolation_xy"):
                panels.append(
                    {
                        "label": f"{model_name} · isolation (x,y)",
                        "svg": info["plot_path_isolation_xy"],
                    }
                )
            if info.get("plot_path_isolation_dy"):
                panels.append(
                    {
                        "label": f"{model_name} · isolation Δy",
                        "svg": info["plot_path_isolation_dy"],
                    }
                )
            slide_entry["models"].append(
                {
                    "model": model_name,
                    "json_path": info["json_path"],
                    "plot_path": info["plot_path"],
                    "plot_path_isolation_xy": info.get("plot_path_isolation_xy"),
                    "plot_path_isolation_dy": info.get("plot_path_isolation_dy"),
                }
            )
        manifest["slides"].append(slide_entry)
        if panels:
            slides_spec.append({"title": slide_title, "panels": panels})

    if args.manifest:
        man_path = os.path.abspath(args.manifest)
        os.makedirs(os.path.dirname(man_path) or ".", exist_ok=True)
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote manifest {man_path}")

    if args.ppt:
        from interactions_presentation import build_multimodel_slides_presentation

        ppt_path = os.path.abspath(args.ppt)
        build_multimodel_slides_presentation(
            slides_spec,
            ppt_path,
            raster_dpi=args.raster_dpi,
        )
        print(f"Wrote presentation {ppt_path} ({len(slides_spec)} slides)")


if __name__ == "__main__":
    main()
