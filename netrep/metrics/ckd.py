import numpy as np
from sklearn.base import BaseEstimator

from netrep.metrics.hsic import HSIC

class CKD:
    """
    Centered Kernel Distance, equivalent to average linear decoding distance
        Given two kernels K, H, CKD(K, H) = sqrt(HSIC(K, K) + HSIC(H, H) - 2*HSIC(K, H)) = ||K - H||_F
        Equivalent to the Euclidean distance between two kernels
    """

    def __init__(self):
        pass

    def fit(self, X, Y):
        pass

    def score(self, X, Y, mode = 'biased', return_squared = False):
        """
        Parameters
        ----------
        X : ndarray
            (num_samples x num_neurons_x) matrix of activations.
        Y : ndarray
            (num_samples x num_neurons_y) matrix of activations.

        mode : str
            particular methods of calculating CKD
                default = 'biased'

            'biased': CKD estimator constructed via biased HSIC estimator from Gretton et al 2005
            'naive_debiased': CKD estimator constructed via debiased HSIC estimator from Song et al 2012, only M-debiased
            'full_debiased': CKD estimator constructed from the U-statistic, debiased for M and N

        return_squared : bool
            whether to return the squared distance
                default = False
            it is recommended to return squared distance for 'full_debiased' mode because debiasing renders many estimates negative

        Returns
        -------
        dist : float
            Distance between X and Y.
        """

        assert X.shape[0] == Y.shape[0], f'Number of samples do not align! (X.shape = {X.shape}, Y.shape = {Y.shape})'

        hsic = HSIC()
        X_at_Y = hsic.score(X, Y, mode = mode)

        if mode == 'biased' or mode == 'naive_debiased':
            # HSIC-based computation of CKA inherits bias/debiased structure of HSIC            
            X_at_X = hsic.score(X, X, mode = mode)
            Y_at_Y = hsic.score(Y, Y, mode = mode)
        elif mode == 'full_debiased':
            X_at_X = hsic.hsic_self(X)
            Y_at_Y = hsic.hsic_self(Y)
        else:
            raise ValueError(f'mode {mode} does not exist')
            
        dist_sq = X_at_X + Y_at_Y - 2*X_at_Y
        if return_squared:
            return dist_sq
        else:
            return np.sqrt(dist_sq)
