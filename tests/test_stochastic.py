"""
Tests stochastic metric.
"""
import pytest
import numpy as np
from netrep.metrics import StochasticMetric
from netrep.utils import angular_distance, rand_orth
from numpy.testing import assert_array_almost_equal, assert_allclose
from sklearn.utils.validation import check_random_state

@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('n_rep', [10])
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('n', [5])
def test_gaussian(seed, n_rep, m, n):
    rs = check_random_state(seed)
    X = rs.randn(n_rep, m, n)
    Y = np.copy(X) @ rand_orth(n)
    metric = StochasticMetric(max_iter=1000)

    metric.fit(X, Y)
    d0 = metric.biased_score(X, Y)

    assert_allclose(
        2 * d0, metric.Xself_ + metric.Yself_,
        atol=1e-3, rtol=1e-2,
    )

