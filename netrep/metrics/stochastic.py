import numpy as np
from netrep.utils import align, sq_bures_metric, rand_orth

class GaussianStochasticMetric:

    def __init__(self, alpha=1.0, group="orth", init="means"):
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

    def fit(self, X, Y, niter=100, tol=1e-6):
        means_X, covs_X = X
        means_Y, covs_Y = Y

        assert means_X.shape == means_Y.shape
        assert covs_X.shape == covs_Y.shape
        assert means_X.shape[0] == covs_X.shape[0]
        assert means_X.shape[1] == covs_X.shape[1]
        assert means_X.shape[1] == covs_X.shape[2]

        vX, uX = np.linalg.eigh(covs_X)
        sX = np.einsum("ijk,ik,ilk->ijl", uX, np.sqrt(vX), uX)
        
        vY, uY = np.linalg.eigh(covs_Y)
        sY = np.einsum("ijk,ik,ilk->ijl", uY, np.sqrt(vY), uY)

        if self.init == "means":
            T = align(means_Y, means_X, group=self.group)
        elif self.init == "rand":
            T = rand_orth(means_X.shape[1])
        loss_hist = []

        for i in range(niter):
            Qs = [align(T.T @ sy, sx, group="orth") for sx, sy in zip(sX, sY)]
            A = np.row_stack(
                [self.alpha * means_X] +
                [(2 - self.alpha) * sx for sx in sX]
            )
            r_sY = []
            B = np.row_stack(
                [self.alpha * means_Y] +
                [Q.T @ ((2 - self.alpha) * sy) for Q, sy in zip(Qs, sY)]
            )
            T = align(B, A, group=self.group)
            loss_hist.append(np.linalg.norm(A - B @ T))
            if i < 2:
                pass
            elif (loss_hist[-2] - loss_hist[-1]) < tol:
                break

        self.T = T
        self.loss_hist = loss_hist

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
        mn = np.mean((self.alpha ** 2) * A + ((2 - self.alpha) ** 2) * B)
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

        m = X.shape[0] * X.shape[1]
        n = X.shape[-1]

        X = X.reshape(m, n)
        Y = Y.reshape(m, n)

        w = np.ones(m)
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
        Xp = np.roll(X, 1, axis=1)
        Yp = np.roll(Y, 1, axis=1)

        E_xy = np.mean(np.linalg.norm(X - Y, axis=-1))
        E_xx = np.mean(np.linalg.norm(X - Xp, axis=-1))
        E_yy = np.mean(np.linalg.norm(Y - Yp, axis=-1))

        return E_xy - .5*(E_xx + E_yy)
