"""Typed domain records shared across CRELAT subsystems."""

from crelat.domain.interaction import PlayFeature, SpeechInteraction
from crelat.domain.experiment import ExperimentManifest
from crelat.domain.play import Play, PlaySpec, Scene
from crelat.domain.speech import Speech

__all__ = [
    "ExperimentManifest",
    "Play",
    "PlayFeature",
    "PlaySpec",
    "Scene",
    "Speech",
    "SpeechInteraction",
]
