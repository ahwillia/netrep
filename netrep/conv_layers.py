import numpy as np
import itertools
from netrep.validation import check_equal_shapes
from tqdm import tqdm


def convolve_metric(metric, X, Y):
    """
    Computes representation metric between convolutional layers,
    convolving activations with boundary conditions.

    Parameters
    ----------
    metric : Metric
        Specifies metric to compute.
    X : ndarray
        Activations from first layer (images x height x width x channel)
    Y : ndarray
        Activations from second layer (images x height x width x channel)

    Returns
    -------
    dists : ndarray
        Matrix with shape (height x width). Holds `metric.score()` for
        X and Y, convolving over the two spatial dimensions.
    """

    # Inputs are (images x height x width x channel) tensors, holding activations.
    X, Y = check_equal_shapes(X, Y, nd=4, zero_pad=metric.zero_pad)
    m, h, w, c = X.shape

    # Flattened Y tensor.
    Yf = Y.reshape(-1, c)

    # Compute metric over all possible offsets.
    pbar = tqdm(total=(w * h))
    dists = np.full((h, w), -1.0)
    for i, j in itertools.product(range(h), range(w)):

        # Apply shift to X tensor, then flatten.
        shifts = (i - (h // 2), j - (w // 2))
        Xf = np.roll(X, shifts, axis=(1, 2)).reshape(-1, c)

        # Fit and evaluate metric.
        metric.fit(Xf, Yf)
        dists[i, j] = metric.score(Xf, Yf)

        # Update progress bar.
        pbar.update()

    pbar.close()
    return dists
