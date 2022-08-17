"""
Tests metrics betwen stochastic neuralrepresentations.
"""
import pytest
import numpy as np
from netrep.metrics import GaussianStochasticMetric, EnergyStochasticMetric
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
    metric.fit(X, Y)
    assert abs(metric.score(X, Y)) < TOL


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [10])
@pytest.mark.parametrize('n', [4])
def test_gaussian_zero_means(seed, m, n):

    # Set random seed, draw random rotation
    rs = check_random_state(seed)
    Q = rand_orth(n, n, random_state=rs)
    Us = [rand_orth(n, n, random_state=np.random.RandomState(2)) for _ in range(m)]
    Ps = [u @ np.diag(np.logspace(-1, 1, n)) @ u.T for u in Us]

    # Create a pair of randomly rotated Gaussians.
    mean_X = np.zeros((m, n))
    mean_Y = np.zeros((m, n))
    covs_X = np.array(Ps)
    covs_Y = np.array([Q.T @ p @ Q for p in Ps])

    X = (mean_X, covs_X)
    Y = (mean_Y, covs_Y)

    # Fit model, assert distance == 0.
    metric = GaussianStochasticMetric(group="orth")
    metric.fit(X, Y)
    assert abs(metric.score(X, Y)) < TOL


@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [10])
@pytest.mark.parametrize('n', [4])
def test_gaussian_general(seed, m, n):

    # Set random seed, draw random rotation
    rs = check_random_state(seed)
    Q = rand_orth(n, n, random_state=rs)
    Us = [rand_orth(n, n, random_state=np.random.RandomState(2)) for _ in range(m)]
    Ps = [u @ np.diag(np.logspace(-1, 1, n)) @ u.T for u in Us]

    # Create a pair of randomly rotated Gaussians.
    mean_X = rs.randn(m, n)
    mean_Y = mean_X @ Q
    covs_X = np.array(Ps)
    covs_Y = np.array([Q.T @ p @ Q for p in Ps])

    X = (mean_X, covs_X)
    Y = (mean_Y, covs_Y)

    # Fit model, assert distance == 0.
    metric = GaussianStochasticMetric(group="orth")
    metric.fit(X, Y)
    assert abs(metric.score(X, Y)) < TOL



@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('m', [4])
@pytest.mark.parametrize('n', [4])
@pytest.mark.parametrize('p', [100])
@pytest.mark.parametrize('noise', [0.1])
def test_energy_distance(seed, m, n, p, noise):

    # Set random seed, draw random rotation
    rs = check_random_state(seed)
    Q = rand_orth(n, n, random_state=rs)

    # Create a pair of randomly rotated Gaussians.
    xm = rs.randn(m, n)
    X = xm[:, None, :] + noise * rs.randn(m, p, n)
    Y = (xm @ Q)[:, None, :] + noise * rs.randn(m, p, n)

    # Fit model.
    metric = EnergyStochasticMetric(group="orth")
    metric.fit(X, Y)

    # Check that loss monotonically decreases.
    assert np.all(np.diff(metric.loss_hist) <= TOL)
