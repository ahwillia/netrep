import itertools
import numpy as np
from netrep.utils import align, sq_bures_metric, rand_orth
from sklearn.utils.validation import check_random_state


class GaussianStochasticMetric:

    def __init__(
            self, alpha=1.0, group="orth", init="means", niter=1000, tol=1e-8,
            random_state=None, n_restarts=1
        ):
        """
        alpha : float between 0 and 2
            When alpha == 0, only uses covariance
            When alpha == 1, equals Wasserstein
            When alpha == 2, only uses means (i.e. deterministic metric)
        """

        if (alpha < 0) or (alpha > 2):
            raise ValueError("alpha parameter should be between zero and two.")
        self.alpha = alpha
        self.group = group
        self.init = init
        self.niter = niter
        self.tol = tol
        self._rs = check_random_state(random_state)
        self.n_restarts = n_restarts
        if self.init == "means":
            assert n_restarts == 1

    def fit(self, X, Y):
        means_X, covs_X = X
        means_Y, covs_Y = Y

        assert means_X.shape == means_Y.shape
        assert covs_X.shape == covs_Y.shape
        assert means_X.shape[0] == covs_X.shape[0]
        assert means_X.shape[1] == covs_X.shape[1]
        assert means_X.shape[1] == covs_X.shape[2]

        best_loss = np.inf
        for restart in range(self.n_restarts):

            if self.init == "means":
                init_T = align(means_Y, means_X, group=self.group)
            elif self.init == "rand":
                init_T = rand_orth(means_X.shape[1], random_state=self._rs)

            T, loss_hist = _fit_gaussian_alignment(
                means_X, means_Y, covs_X, covs_Y, init_T,
                self.alpha, self.group, self.niter, self.tol
            )
            if best_loss > loss_hist[-1]:
                best_loss = loss_hist[-1]
                best_T = T

        self.T = best_T
        self.loss_hist = loss_hist
        return self

    def transform(self, X, Y):
        means_Y, covs_Y = Y
        return X, (
            means_Y @ self.T,
            np.einsum("ijk,jl,kp->ilp", covs_Y, self.T, self.T)
        )

    def score(self, X, Y):
        X, Y = self.transform(X, Y)
        mX, sX = X
        mY, sY = Y

        A = np.sum((mX - mY) ** 2, axis=1)
        B = np.array([sq_bures_metric(sx, sy) for sx, sy in zip(sX, sY)])
        mn = np.mean(self.alpha * A + (2 - self.alpha) * B)
        # mn should always be positive but sometimes numerical rounding errors
        # cause mn to be very slightly negative, causing sqrt(mn) to be nan.
        # Thus, we take sqrt(abs(mn)) and pass through the sign. Any large
        # negative outputs should be caught by unit tests.
        return np.sign(mn) * np.sqrt(abs(mn))


class EnergyStochasticMetric:

    def __init__(self, group="orth"):
        self.group = group

    def fit(self, X, Y, niter=100, tol=1e-6):
        # X.shape = (images x repeats x neurons)
        # Y.shape = (images x repeats x neurons)

        assert X.shape == Y.shape

        r = X.shape[1]

        # m = X.shape[0] * X.shape[1]
        # n = X.shape[-1]
        # X = X.reshape(m, n)
        # Y = Y.reshape(m, n)
        idx = np.array(list(itertools.product(range(r), range(r))))
        X = np.row_stack([x[idx[:, 0]] for x in X])
        Y = np.row_stack([y[idx[:, 1]] for y in Y])

        w = np.ones(X.shape[0])
        loss_hist = [np.mean(np.linalg.norm(X - Y, axis=-1))]

        for i in range(niter):
            Q = align(w[:, None] * Y, w[:, None] * X, group=self.group)
            resid = np.linalg.norm(X - Y @ Q, axis=-1)
            loss_hist.append(np.mean(resid))
            w = 1 / np.maximum(np.sqrt(resid), 1e-6)
            if (loss_hist[-2] - loss_hist[-1]) < tol:
                break

        self.w = w
        self.Q = Q
        self.loss_hist = loss_hist

    def transform(self, X, Y):
        # X.shape = (images x repeats x neurons)
        # Y.shape = (images x repeats x neurons)
        assert X.shape == Y.shape

        return X, np.einsum("ijk,kl->ijl", Y, self.Q)

    def score(self, X, Y):
        X, Y = self.transform(X, Y)
        m = X.shape[0] # num images
        n_samples = X.shape[1]

        combs = np.array(list(
            itertools.combinations(range(n_samples), 2)
        ))
        prod = np.array(list(
            itertools.product(range(n_samples), range(n_samples))
        ))
        
        d_xy, d_xx, d_yy = 0, 0, 0
        for i in range(m):
            d_xy += np.mean(np.linalg.norm(X[i][prod[:, 0]] - Y[i][prod[:, 1]], axis=-1))
            d_xx += np.mean(np.linalg.norm(X[i][combs[:, 0]] - X[i][combs[:, 1]], axis=-1))
            d_yy += np.mean(np.linalg.norm(Y[i][combs[:, 0]] - Y[i][combs[:, 1]], axis=-1))

        return (d_xy / m) - .5*((d_xx / m) + (d_yy / m))



def _fit_gaussian_alignment(
        means_X, means_Y, covs_X, covs_Y, T, alpha, group, niter, tol
    ):
    vX, uX = np.linalg.eigh(covs_X)
    sX = np.einsum("ijk,ik,ilk->ijl", uX, np.sqrt(vX), uX)
    
    vY, uY = np.linalg.eigh(covs_Y)
    sY = np.einsum("ijk,ik,ilk->ijl", uY, np.sqrt(vY), uY)

    loss_hist = []

    for i in range(niter):
        Qs = [align(T.T @ sy, sx, group="orth") for sx, sy in zip(sX, sY)]
        A = np.row_stack(
            [alpha * means_X] +
            [(2 - alpha) * sx for sx in sX]
        )
        r_sY = []
        B = np.row_stack(
            [alpha * means_Y] +
            [Q.T @ ((2 - alpha) * sy) for Q, sy in zip(Qs, sY)]
        )
        T = align(B, A, group=group)
        loss_hist.append(np.linalg.norm(A - B @ T))
        if i < 2:
            pass
        elif (loss_hist[-2] - loss_hist[-1]) < tol:
            break

    return T, loss_hist
