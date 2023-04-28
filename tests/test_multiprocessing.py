import os
os.environ['OMP_NUM_THREADS'] = '1'

import pytest
import numpy as np
from netrep.metrics import LinearMetric, PermutationMetric
from netrep.metrics import GaussianStochasticMetric, EnergyStochasticMetric
from sklearn.utils.validation import check_random_state


def _get_cov(n_images, n_neurons, rs):
    A = rs.randn(n_images, n_neurons, n_neurons)
    # batched outerproduct
    return np.einsum('bij,bkj->bik', A, A)


def _get_data_wasserstein(n_networks, n_images, n_neurons, rs):
    return [(rs.randn(n_images, n_neurons), _get_cov(n_images, n_neurons, rs)) for _ in range(n_networks)]


@pytest.mark.parametrize('metric_type', ['linear', 'permutation', 'gaussian', 'energy'])
@pytest.mark.parametrize('test_set', [False, True])
def test_pairwise_distances(metric_type, test_set):

    rs = check_random_state(0)
    
    # Create a pair of randomly rotated matrices.
    n_networks, n_images, n_repeats, n_neurons = 3, 2, 3, 4

    if metric_type == 'linear':
        train_data = [rs.randn(n_images, n_neurons) for _ in range(n_networks)]
        test_data = [rs.randn(n_images, n_neurons) for _ in range(n_networks)]

        metric = LinearMetric()

    elif metric_type == 'permutation':
        train_data = [rs.randn(n_images, n_neurons) for _ in range(n_networks)]
        test_data = [rs.randn(n_images, n_neurons) for _ in range(n_networks)]

        metric = PermutationMetric()

    if metric_type == 'energy':
        metric = EnergyStochasticMetric()
        train_data = [rs.randn(n_images, n_repeats, n_neurons) for _ in range(n_networks)]
        test_data = [rs.randn(n_images, n_repeats, n_neurons) for _ in range(n_networks)]

    elif metric_type == 'gaussian':
        metric = GaussianStochasticMetric()
        train_data = _get_data_wasserstein(n_networks, n_images, n_neurons, rs)
        test_data = _get_data_wasserstein(n_networks, n_images, n_neurons, rs)

    if test_set:
        D_train, D_test = metric.pairwise_distances(train_data, test_data)
        assert D_test.sum() >= 0.0
    else:
        D_train, D_test = metric.pairwise_distances(train_data)
        assert D_test.sum() == np.inf

    assert D_train.shape == (n_networks, n_networks)
    assert D_test.shape == (n_networks, n_networks)
    assert D_train.sum() >= 0.0
