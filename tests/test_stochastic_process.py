# %%
"""
Tests metrics betwen stochastic process neural representations.
"""
import numpy as np
from netrep.metrics import GPStochasticMetric,GaussianStochasticMetric
from netrep.utils import rand_orth
from sklearn.utils.validation import check_random_state
from sklearn.covariance import EmpiricalCovariance

from numpy import random as rand
from scipy.stats import ortho_group

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

# %% Generating data from a GP given number of dimensions and time points
n_dims = 3
n_times = 10
n_samples = 500

ts = np.linspace(0,1,n_times)
gpA = GaussianProcess(
    kernel = lambda x, y: 1e-2*(1e-6*(x==y)+np.exp(-np.linalg.norm(x-y)**2/(2*1.**2))),
    D=n_dims
)

sA = np.array([gpA.sample(ts) for _ in range(n_samples)])
sAvec = sA.reshape(n_samples,n_dims*n_times)

# %% Transforming the GP using an orthonormal matrix and computing SSD and DSSD
Q = ortho_group.rvs(dim=n_dims)
print('Determinant: ', np.linalg.det(Q))

metric = GPStochasticMetric(n_dims=n_dims,group="orth")
A = [sAvec.mean(0),EmpiricalCovariance().fit(sAvec).covariance_]

B = [
    np.kron(np.eye(n_times),Q)@A[0], 
    np.kron(np.eye(n_times),Q)@A[1]@(np.kron(np.eye(n_times),Q)).T
]

metric.fit_score(A,B)
# %% Computing SSD on the full NT*NT means and covariances
metric = GaussianStochasticMetric(group="orth")

A = [A[0][None],A[1][None]]
B = [B[0][None],B[1][None]]

metric.fit_score(A,B)
# %% DSSD between GPs as a function of length scale
lambdas = np.linspace(.1,1,5)
dssds = np.zeros((len(lambdas),len(lambdas)))
for i in range(len(lambdas)):
    for j in range(len(lambdas)):
    
        gpA = GaussianProcess(
            kernel = lambda x, y: 1e-2*(1e-6*(x==y)+np.exp(-np.linalg.norm(x-y)**2/(2*lambdas[i]**2))),
            D=n_dims
        )

        gpB = GaussianProcess(
            kernel = lambda x, y: 1e-2*(1e-6*(x==y)+np.exp(-np.linalg.norm(x-y)**2/(2*lambdas[j]*2))),
            D=n_dims
        )

        sA = np.array([gpA.sample(ts) for _ in range(n_samples)]).reshape(n_samples,n_dims*n_times)
        sB = np.array([gpB.sample(ts) for _ in range(n_samples)]).reshape(n_samples,n_dims*n_times)

        A = [sA.mean(0),EmpiricalCovariance().fit(sA).covariance_]
        B = [sB.mean(0),EmpiricalCovariance().fit(sB).covariance_]

        metric = GPStochasticMetric(n_dims=n_dims,group="orth")
        dssds[i,j] = metric.fit_score(A,B)

# %% Plotting the result
import matplotlib.pyplot as plt
plt.imshow(dssds)
plt.xlabel('$\lambda$')
plt.ylabel('$\lambda$')
plt.xticks(np.arange(len(lambdas)),lambdas)
plt.yticks(np.arange(len(lambdas)),lambdas)
plt.colorbar()
plt.show()
