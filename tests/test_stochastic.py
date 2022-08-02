"""
Tests metrics betwen stochastic neuralrepresentations.
"""
import pytest
import numpy as np
from netrep.metrics import GaussianStochasticMetric
from netrep.utils import rand_orth
from sklearn.utils.validation import check_random_state

TOL = 1e-6


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [10])
@pytest.mark.parametrize('n', [4])
def test_gaussian_identity_covs(seed, m, n):

    # Set random seed, draw random rotation
    rs = check_random_state(seed)
    Q = rand_orth(n, n, random_state=rs)

    # Create a pair of randomly rotated Gaussians.
    mean_X = rs.randn(m, n)
    mean_Y = mean_X @ Q
    covs_X = np.array([np.eye(n) for _ in range(m)])
    covs_Y = np.array([np.eye(n) for _ in range(m)])

    X = (mean_X, covs_X)
    Y = (mean_Y, covs_Y)

    # Fit model, assert distance == 0.
    metric = GaussianStochasticMetric(group="orth")
    metric.fit(X, Y, niter=0)
    assert abs(metric.score(X, Y)) < TOL

