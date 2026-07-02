"""Interaction and feature records."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechInteraction:
    play_id: str
    play_title: str
    scene_id: str
    speech1_id: str
    speech2_id: str
    speaker1: str
    speaker2: str
    text1: str
    text2: str
    cosine_similarity: float
    model_id: str


@dataclass(frozen=True)
class PlayFeature:
    play_id: str
    representation: str
    feature: str
    value: float
    transform: str = "identity"
