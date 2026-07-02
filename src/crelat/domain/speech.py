"""Speech-level domain records."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Speech:
    id: str
    play_id: str
    scene_id: str
    position: int
    speaker_id: str
    text: str
