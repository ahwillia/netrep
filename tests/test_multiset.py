"""
Tests metrics between network representations.
"""
import pytest
import numpy as np
import netrep.metrics
from netrep.multiset import pairwise_distances
from sklearn.utils.validation import check_random_state
from numpy.testing import assert_allclose

TOL = 1e-6


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('metricname', ['LinearMetric', 'PermutationMetric'])
@pytest.mark.parametrize('m', [100])
@pytest.mark.parametrize('n', [10])
@pytest.mark.parametrize('num_X', [4])
def test_pairwise_caching(seed, metricname, m, n, num_X):

    # Set random seed, draw random rotation.
    rs = check_random_state(seed)
    Xs = [rs.randn(m, n) for _ in range(num_X)]

    # Specify metric.
    metric = getattr(netrep.metrics, metricname)()

    # Compute pairwise distances with caching.
    D1 = pairwise_distances(
        metric, Xs,
        verbose=False, enable_caching=True
    )

    # Compute pairwise distances without caching.
    D2 = pairwise_distances(
        metric, Xs,
        verbose=False, enable_caching=False
    )
    assert np.max(np.abs(D1 - D2)) < TOL

