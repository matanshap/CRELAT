"""Model-independent embedding protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class TextEmbedder(ABC):
    model_id: str
    output_dimension: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one embedding row per input text."""
