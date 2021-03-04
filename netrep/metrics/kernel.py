# class KernelizedMetric(BaseEstimator, MetricMixin):

#     def __init__(
#             self, alpha=1.0, gamma=0.0, method="exact",
#             kernel="linear", kernel_params=dict(), zero_pad=True):

#         if (alpha > 1) or (alpha < 0):
#             raise ValueError(
#                 "Regularization parameter `alpha` must be between zero and one.")

#         if (gamma > 1) or (gamma < 0):
#             raise ValueError(
#                 "Regularization parameter `gamma` must be between zero and one.")

#         if ((alpha + gamma) > 1) or ((alpha + gamma) < 0):
#             raise ValueError(
#                 "Regularization parameters `alpha` and `gamma` must sum to a "
#                 "number between zero and one.")

#         self.alpha = alpha
#         self.gamma = gamma
#         self.method = "exact"
#         self.kernel = kernel
#         self.kernel_params = kernel_params
#         self.zero_pad = zero_pad

#         if ("metric" in kernel_params) and (kernel_params["metric"] != kernel):
#             raise ValueError(
#                 "If 'metric' keyword is included in 'kernel_params' "
#                 "it must match 'kernel' parameter.")
#         else:
#             self.kernel_params["metric"] = self.kernel


#     def fit(self, X, Y):

#         X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)
#         n_obs, n_feats = X.shape

#         if self.method == "exact":

#             # Compute kernel matrices.
#             Kx = centered_kernel(X, **self.kernel_params)
#             Ky = centered_kernel(Y, **self.kernel_params)

#             # Whiten kernel matrices.
#             Kx_whitened, Zx = _whiten_kernel_matrix(Kx, self.alpha, self.gamma)
#             Ky_whitened, Zy = _whiten_kernel_matrix(Ky, self.alpha, self.gamma)

#             # Multiply kernel matrices, compute SVD.
#             U, _, Vt = np.linalg.svd(Kx_whitened @ Ky_whitened)

#             # Compute alignment transformations.
#             self.Wx_ = Zx @ U
#             self.Wy_ = Zy @ Vt.T

#             # Store training set, for prediction at test time.
#             self.X_ = X.copy()
#             self.Y_ = Y.copy()

#         elif self.method in ("rand", "randomized"):

#             # Approximate low-rank eigendecompositions
#             lam_x, Vx = randomized_kernel_eigh(X, self.kernel_params)
#             lam_y, Vy = randomized_kernel_eigh(Y, self.kernel_params)

#             wx = np.full(n_obs, self.gamma)
#             wx[:lam_x.size] += (1 - self.alpha - self.gamma) * (lam_x ** 2)
#             wx[:lam_x.size] += (self.alpha * lam_x)

#             wx = np.full(n_obs, self.gamma)
#             wx[:lam_x.size] += (1 - self.alpha - self.gamma) * (lam_x ** 2)
#             wx[:lam_x.size] += (self.alpha * lam_x)

#             assert False

#         return self

#     def transform_X(self, X):
#         check_is_fitted(self, attributes=["Wx_"])
#         return centered_kernel(X, self.X_) @ self.Wx_

#     def transform_Y(self, Y):
#         check_is_fitted(self, attributes=["Wy_"])
#         return centered_kernel(Y, self.Y_) @ self.Wy_

#     def score(self, X, Y):
#         return angular_distance(*self.transform(X, Y))


# def _whiten_kernel_matrix(K, a, g, eigval_tol=1e-7):

#     # Compute eigendecomposition for kernel matrix
#     w, v = np.linalg.eigh(K)

#     # Regularize eigenvalues.
#     w = ((1 - a - g) * (w ** 2)) + (a * w) + g
#     w[w < eigval_tol] = eigval_tol  # clip minimum eigenvalue

#     # Matrix holding the whitening transformation.
#     Z = (v * (1 / np.sqrt(w))[None, :]) @ v.T

#     # Returned (partially) whitened data and whitening matrix.
#     return K @ Z, Z



# def randomized_kernel_approx(X, kernel_params, s, upsample_factor, random_state):
    
#     # Sample s columns of the kernel matrix, randomly at uniform.
#     i1 = random_state.choice(len(X), replace=False, size=s)
#     C = pairwise_kernels(X, X[i1], **kernel_params)

#     # Find an orthonormal basis for the sampled columns.
#     Q, _ = np.linalg.qr(C)

#     # Sample upsample_factor * s columns, using leverage scores
#     lev_scores = np.sum(Q * Q, axis=1)
#     i2 = random_state.choice(
#         len(X)
#         size=(upsample_factor * s),
#         p=(lev_scores / np.sum(lev_scores)),
#         replace=False
#     )

#     # Empirically, including the initially sampled columns helps performance.
#     idx = np.unique(np.concatenate((i1, i2)))

#     # Form low rank estimate, L @ L.T, of kernel matrix.
#     Ksub = pairwise_kernels(X[idx], **kernel_params)
#     L = np.linalg.pinv(Q[idx]) @ scipy.linalg.sqrtm(Ksub)

#     # Compute SVD of L to estimate the eigendecomposition of kernel matrix.
#     eigvecs, sqrt_eigvals, _ = np.linalg.svd(L, full_matrices=False)

#     return sqrt_eigvals ** 2, eigvecs
