import numpy as np
from sklearn.utils.validation import check_random_state
from scipy.linalg import sqrtm

def line_2d_data(
        n_classes=5,
        n_samples=100,
        s1=1.0,
        s2=1.0,
        corr=0.0,
        cov_diff_scale=1.0,
        random_state=None,
    ):

    means = np.row_stack(
        [np.ones(2) * c for c in range(n_classes)]
    )

    cov_base = np.array([
        [s1, corr * np.sqrt(s1 * s2)],
        [corr * np.sqrt(s1 * s2), s2]
    ])
    covs = np.stack(
        [s * cov_base for s in np.linspace(1, cov_diff_scale, n_classes)]
    )

    rs = check_random_state(random_state)
    samples = np.tile(means[:, None, :], (1, n_samples, 1))
    for i in range(n_classes):
        samples[i] += rs.randn(n_samples, 2) @ sqrtm(covs[i])

    return means, covs, samples

