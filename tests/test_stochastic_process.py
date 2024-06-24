# %%
"""
Tests metrics betwen stochastic process neural representations.
"""

import pytest
import numpy as np
from netrep.metrics import GPStochasticMetric,GaussianStochasticMetric
from netrep.utils import rand_orth
from sklearn.utils.validation import check_random_state
from sklearn.covariance import EmpiricalCovariance

from numpy import random as rand
from netrep.utils import rand_orth

TOL = 1e-6

# %% Class for sampling from a gaussian process given a kernel
class GaussianProcess:
    def __init__(self,kernel,D):
        self.kernel = kernel
        self.D = D

    def evaluate_kernel(self, xs, ys):
        fun = np.vectorize(self.kernel)
        return fun(xs[:, None], ys)

    def sample(self,ts):
        T = ts.shape[0]
        c_g = self.evaluate_kernel(ts,ts)
        fs = rand.multivariate_normal(
            mean=np.zeros(T),
            cov=c_g,
            size=self.D
        )
        return fs


# %%
@pytest.mark.parametrize('seed', [1, 2, 3])
@pytest.mark.parametrize('t', [10])  # number of time points
@pytest.mark.parametrize('n', [4])   # number of neurons
@pytest.mark.parametrize('k', [100]) # number of samples
def test_gaussian_process(seed, t, n, k):
    # Set random seed, draw random rotation
    rs = check_random_state(seed)
    Q = rand_orth(n, n, random_state=rs)

    # Generate data from a gaussian process with RBF kernel
    ts = np.linspace(0,1,t)
    gpA = GaussianProcess(
        kernel = lambda x, y: 1e-2*(1e-6*(x==y)+np.exp(-np.linalg.norm(x-y)**2/(2*1.**2))),
        D=n
    )
    sA = np.array([gpA.sample(ts) for _ in range(k)]).reshape(k,n*t)
    
    # Transform GP according to a rotation applied to individiual 
    # blocks of the full covariance matrix
    A = [sA.mean(0),EmpiricalCovariance().fit(sA).covariance_]
    B = [
        np.kron(np.eye(t),Q)@A[0], 
        np.kron(np.eye(t),Q)@A[1]@(np.kron(np.eye(t),Q)).T
    ]
    

    # Compute DSSD
    metric = GPStochasticMetric(n_dims=n,group="orth")

    dssd = metric.fit_score(A,B)
    assert abs(dssd) < TOL

    # Compute marginal SSD
    metric = GaussianStochasticMetric(group="orth")

    A_marginal = [
        A[0].reshape(t,n),
        np.array([A[1][i*n:(i+1)*n,i*n:(i+1)*n] for i in range(t)])
    ]

    B_marginal = [
        B[0].reshape(t,n),
        np.array([B[1][i*n:(i+1)*n,i*n:(i+1)*n] for i in range(t)])
    ]

    marginal_ssd = metric.fit_score(A_marginal,B_marginal)
    assert abs(marginal_ssd) < TOL

    # Compute full SSD
    metric = GaussianStochasticMetric(group="orth")

    A_full = [A[0][None],A[1][None]]
    B_full = [B[0][None],B[1][None]]

    full_ssd = metric.fit_score(A_full,B_full)

    assert abs(full_ssd) > TOL
