"""
Helper functions to check model inputs.
"""

import numpy as np
import numpy.typing as npt
from sklearn.utils.validation import check_array


def check_equal_shapes(
    X: npt.NDArray, 
    Y: npt.NDArray, 
    nd: int = 2, 
    zero_pad: bool = False
    ) -> tuple[npt.NDArray, npt.NDArray]:
    """Checks that X and Y have equal shapes."""

    X = check_array(X, allow_nd=True)
    Y = check_array(Y, allow_nd=True)

    if (X.ndim != nd) or (Y.ndim != nd):
        raise ValueError(
            "Expected {}d arrays, but shapes were {} and "
            "{}.".format(nd, X.shape, Y.shape)
        )

    if X.shape != Y.shape:

        if zero_pad and (X.shape[:-1] == Y.shape[:-1]):
            
            # Number of padded zeros to add.
            n = max(X.shape[-1], Y.shape[-1])
            
            # Padding specifications for X and Y.
            px = np.zeros((nd, 2), dtype="int")
            py = np.zeros((nd, 2), dtype="int")
            px[-1, -1] = n - X.shape[-1]
            py[-1, -1] = n - Y.shape[-1]

            # Pad X and Y with zeros along final axis.
            X = np.pad(X, px)
            Y = np.pad(Y, py)

        else:
            raise ValueError(
                "Expected arrays with equal dimensions, "
                "but got arrays with shapes {} and {}."
                "".format(X.shape, Y.shape))

    return X, Y
