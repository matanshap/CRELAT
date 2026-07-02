"""Play-level domain records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from crelat.domain.speech import Speech


@dataclass(frozen=True)
class PlaySpec:
    id: str
    folger_code: str
    title: str
    genre: str
    year: int
    xml: Path


@dataclass
class Scene:
    id: str
    act_number: Optional[int]
    scene_number: int
    speeches: list[Speech] = field(default_factory=list)


@dataclass
class Play:
    id: str
    title: str
    characters: tuple[str, ...]
    scenes: list[Scene]

    @property
    def speeches(self) -> list[Speech]:
        return [speech for scene in self.scenes for speech in scene.speeches]
