import numpy as np

from crelat.analysis.isolation import mean_absolute_isolation


def test_mean_absolute_isolation_ignores_self():
    np.testing.assert_allclose(mean_absolute_isolation(np.array([0.0, 2.0, 5.0])), [3.5, 2.5, 4.0])


def test_mean_absolute_isolation_preserves_missing_positions():
    result = mean_absolute_isolation(np.array([1.0, np.nan, 4.0]))
    np.testing.assert_allclose(result[[0, 2]], [3.0, 3.0])
    assert np.isnan(result[1])
