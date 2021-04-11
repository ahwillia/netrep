"""
This file implements a metric between stochastic layers
based on the Energy distance

    https://en.wikipedia.org/wiki/Energy_distance
"""

import itertools
import numpy as np
import torch
from torch.optim import LBFGS
from torch.functional import F
from torch import nn
import geotorch


class StochasticMetric:

    def __init__(
            self, center_columns=True, convergence_tol=1e-6,
            max_iter=100, verbose=False, rtol=0.1, atol=1e-3
        ):
        self.center_columns = center_columns
        self.convergence_tol = convergence_tol
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, Y):
        """
        Align stochastic representations X and Y.

        Parameters
        ----------
        X : ndarray
            (num_repeats x num_inputs x num_neurons)
        Y : ndarray
            (num_repeats x num_inputs x num_neurons)
        """

        # Fit alignment parameters.
        Xm, Ym = _flatten_all_pairs(X, Y)
        self.Q_, self.bias_ = _find_rotation_lbfgs(
            Xm, Ym, center_columns=self.center_columns,
            tol=self.convergence_tol, max_iter=self.max_iter, verbose=self.verbose
        )

        # Fit self-distance terms for debiasing
        self.Xself_ = _self_dist(X)
        self.Yself_ = _self_dist(Y)

        return self

    def score(self, X, Y):
        d2 = 2 * self.biased_score(X, Y) - self.Xself_ - self.Yself_
        if (d2 < 0):
            a_badness = np.sqrt(-d2)
            r_badness = -d2 / (self.Xself_ + self.Yself_)
            if (a_badness > self.atol) and (r_badness > self.rtol):
                raise RuntimeError(
                    f"Computed a negative distance. "
                    f"(absolute error > absolute tol: {a_badness} > {self.atol})."
                    f"(relative error > relative tol: {r_badness} > {self.rtol})."
                )
        return np.sqrt(max(0, d2))

    def biased_score(self, X, Y):
        """
        Compute the inflated distance score.
        """
        Xm, Ym = _flatten_all_pairs(X, Y)
        resid = Xm - (Ym @ self.Q_) - self.bias_[None, :]
        return np.mean(np.linalg.norm(resid, axis=1))


def _flatten_all_pairs(X, Y):
    """
    Flatten activation tensors into matrices, concatenating all pairwise differences
    across repeats of the same input.
    """
    n_rep, m, n = X.shape
    Xm, Ym = [], []
    # itr = itertools.product(range(n_rep), range(n_rep))
    itr = itertools.combinations(range(n_rep), 2)
    for (i1, i2), j in itertools.product(itr, range(m)):
        Xm.append(X[i1, j])
        Ym.append(Y[i2, j])

    # Convert into numpy arrays.
    return np.array(Xm), np.array(Ym)


def _self_dist(X):
    """
    Flatten X and into a matrix, concatenating combinations without
    replacement for the repeats.
    """
    n_rep, m, n = X.shape
    itr = itertools.combinations(range(n_rep), 2)
    d = 0.0
    c = 0
    for (i1, i2), j in itertools.product(itr, range(m)):
        d += np.linalg.norm(X[i1, j] - X[i2, j])
        c += 1
    return d / c
    # return d / (m * n_rep * (n_rep - 1) // 2)


def _find_rotation_lbfgs(
        X, Y, tol=1e-6, max_iter=100, verbose=True, center_columns=True,
    ):
    """
    Finds orthogonal matrix Q, scaling s, and translation b, to

        minimize   sum(norm(X - s * Y @ Q - b)).

    Note that the solution is not in closed form because we are
    minimizing the sum of norms, which is non-trivial given the
    orthogonality constraint on Q. Without the orthogonality
    constraint, the problem can be formulated as a cone program:

        Guoliang Xue & Yinyu Ye (2000). "An Efficient Algorithm for
        Minimizing a Sum of p-Norms." SIAM J. Optim., 10(2), 551–579.

    However, the orthogonality constraint complicates things, so
    we just minimize by gradient methods used in manifold optimization.

        Mario Lezcano-Casado (2019). "Trivializations for gradient-based
        optimization on manifolds." NeurIPS.
    """

    # Convert X and Y to pytorch tensors.
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    # Check inputs.
    m, n = X.shape
    assert Y.shape == X.shape

    # Orthogonal linear transformation.
    Q = nn.Linear(n, n, bias=False)
    geotorch.orthogonal(Q, "weight")
    Q = Q.double()

    # Allow a rigid translation.
    bias = nn.Parameter(torch.zeros(n, dtype=torch.float64))

    # Collect trainable parameters
    trainable_params = list(Q.parameters())

    if center_columns:
        trainable_params.append(bias)

    # Define rotational alignment, and optimizer.
    optimizer = LBFGS(
        trainable_params,
        max_iter=100, # number of inner iterations.
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = torch.mean(
            torch.norm(X - Q(Y) - bias, dim=1)
        )
        loss.backward()
        return loss

    # Fit parameters.
    converged = False
    itercount = 0
    while (not converged) and (itercount < max_iter):

        # Update parameters.
        new_loss = optimizer.step(closure).item()

        # Check convergence.
        if itercount != 0:
            improvement = (last_loss - new_loss) / last_loss
            converged = improvement < tol
        
        last_loss = new_loss

        # Display progress.
        itercount += 1
        if verbose:
            print(f"Iter {itercount}: {last_loss}")
            if converged:
                print("Converged!")

    # Extract result in numpy.
    Q_ = Q.weight.detach().numpy()
    bias_ = bias.detach().numpy()

    return Q_, bias_
