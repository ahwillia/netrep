from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_random_state, check_is_fitted

from netrep.utils import rand_orth
import numpy as np


class RBFOrthoSampler(TransformerMixin, BaseEstimator):

    def __init__(self, gamma=1., n_components=100, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):

        X = self._validate_data(X, accept_sparse='csr')
        rs = check_random_state(self.random_state)
        n_features = X.shape[1]

        nc = self.n_components // 2
        self.random_weights_ = np.full((n_features, nc), np.nan)

        i = 0
        while i < nc:
            Q = rand_orth(n_features, random_state=rs)
            j = min(nc, i + n_features)
            self.random_weights_[:, i:j] = Q[:, :(j - i)]
            i = j

        self.random_weights_ *= np.sqrt(2 * self.gamma)
        self.random_weights_ *= np.sqrt(rs.chisquare(n_features, size=(1, nc)))

        return self

    def transform(self, X):
        check_is_fitted(self)
        P = X @ self.random_weights_
        return np.column_stack((np.cos(P), np.sin(P))) / np.sqrt(P.shape[1])
