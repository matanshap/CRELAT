"""Research analyses over canonical feature tables."""

from crelat.analysis.genre import compute_genre_features
from crelat.analysis.isolation import mean_absolute_isolation

__all__ = ["compute_genre_features", "mean_absolute_isolation"]
