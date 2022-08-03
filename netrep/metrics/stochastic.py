import numpy as np
from netrep.utils import align, sq_bures_metric

class GaussianStochasticMetric:

    def __init__(self, group="orth"):
        self.group = group

    def fit(self, X, Y, niter=100):
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

        T = align(means_Y, means_X, group=self.group)
        loss_hist = []

        for i in range(niter):
            Qs = [align(T.T @ sy, sx, group="orth") for sx, sy in zip(sX, sY)]
            A = np.row_stack(
                [means_X] + [sx for sx in sX]
            )
            r_sY = []
            B = np.row_stack(
                [means_Y] + [Q.T @ sy for Q, sy in zip(Qs, sY)]
            )
            T = align(B, A, group=self.group)
            loss_hist.append(np.linalg.norm(A - B @ T))

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
        mn = np.mean(A + B)
        # mn should always be positive but sometimes numerical rounding errors
        # cause mn to be very slightly negative, causing sqrt(mn) to be nan.
        # Thus, we take sqrt(abs(mn)) and pass through the sign. Any large
        # negative outputs should be caught by unit tests.
        return np.sign(mn) * np.sqrt(abs(mn))



# means_X = np.random.randn(5, 3)
# covs_X = np.array([np.eye(3) for _ in range(5)])
# means_Y = np.random.randn(5, 3)
# covs_Y = np.array([np.eye(3) for _ in range(5)])
# X = (means_X, covs_X)
# Y = (means_Y, covs_X)

# metric = GaussianStochasticMetric()
# metric.fit(X, Y)
# print(metric.T)
# print(align(means_X, means_Y))

class SinkhornStochasticMetric:
    pass # todo


