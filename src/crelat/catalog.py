"""Load and validate the canonical Shakespeare play catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import yaml

from crelat.domain.play import PlaySpec
from crelat.paths import DATA_DIR, resolve_project_path


DEFAULT_CATALOG = DATA_DIR / "catalog" / "plays.yaml"
VALID_GENRES = {"comedy", "history", "tragedy"}


def load_play_catalog(
    path: Union[str, Path] = DEFAULT_CATALOG, *, require_files: bool = False
) -> list[PlaySpec]:
    catalog_path = resolve_project_path(path)
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    rows = payload.get("plays")
    if not isinstance(rows, list):
        raise ValueError(f"Catalog {catalog_path} must contain a 'plays' list")

    plays = [
        PlaySpec(
            id=str(row["id"]),
            folger_code=str(row["folger_code"]),
            title=str(row["title"]),
            genre=str(row["genre"]),
            year=int(row["year"]),
            xml=resolve_project_path(row["xml"]),
        )
        for row in rows
    ]
    _validate_catalog(plays, require_files=require_files)
    return plays


def _validate_catalog(plays: Iterable[PlaySpec], *, require_files: bool) -> None:
    rows = list(plays)
    for field in ("id", "folger_code", "title"):
        values = [getattr(play, field).casefold() for play in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"Play catalog contains duplicate {field} values")
    invalid = sorted({play.genre for play in rows} - VALID_GENRES)
    if invalid:
        raise ValueError(f"Unknown genres: {', '.join(invalid)}")
    if require_files:
        missing = [str(play.xml) for play in rows if not play.xml.is_file()]
        if missing:
            raise FileNotFoundError("Missing play XML files:\n" + "\n".join(missing))


def catalog_by_id(path: Union[str, Path] = DEFAULT_CATALOG) -> dict[str, PlaySpec]:
    return {play.id: play for play in load_play_catalog(path)}
