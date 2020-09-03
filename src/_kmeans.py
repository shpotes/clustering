from functools import partial
import jax
import jax.numpy as jnp
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

@partial(jax.jit, static_argnums=(2,))
def _vector_quantize(points, codebook, norm_ord):
    norm = partial(jnp.linalg.norm, ord=norm_ord)
    assignment = jax.vmap(
        lambda point: jnp.argmin(jax.vmap(norm)(codebook - point))
    )(points)
    distns = jax.vmap(norm)(codebook[assignment,:] - points)
    return assignment, distns

@partial(jax.jit, static_argnums=(2,3,))
def _kmeans_run(key, points, k, norm_ord, thresh=1e-5):

    def improve_centroids(val):
        prev_centroids, prev_distn, _ = val
        assignment, distortions = _vector_quantize(points, prev_centroids, norm_ord)

        # Count number of points assigned per centroid
        counts = (
            (assignment[jnp.newaxis, :] == jnp.arange(k)[:, jnp.newaxis])
            .sum(axis=1, keepdims=True)
            .clip(a_min=1.)  # clip to change 0/0 later to 0/1
        )

        # Sum over points in a centroid by zeroing others out
        new_centroids = jnp.sum(
            jnp.where(
                assignment[:, jnp.newaxis, jnp.newaxis] \
                    == jnp.arange(k)[jnp.newaxis, :, jnp.newaxis],
                points[:, jnp.newaxis, :],
                0.,
            ),
            axis=0,
        ) / counts

        new_centroids /= counts

        return new_centroids, jnp.mean(distortions), prev_distn

    initial_indices = jax.random.permutation(key, jnp.arange(points.shape[0]))[:k]
    initial_val = improve_centroids((points[initial_indices, :], jnp.inf, None))

    centroids, distortion, _ = jax.lax.while_loop(
        lambda val: (val[2] - val[1]) > thresh,
        improve_centroids,
        initial_val,
    )
    return centroids, distortion

@partial(jax.jit, static_argnums=(2, 3, 4))
def _kmeans(key, points, k, norm_ord, restarts, **kwargs):
    all_centroids, all_distortions = jax.vmap(
        lambda key: _kmeans_run(key, points, k, norm_ord, **kwargs)
    )(jax.random.split(key, restarts))
    i = jnp.argmin(all_distortions)
    return all_centroids[i], all_distortions[i]

class KMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters=8, norm_ord=2, tol=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self._norm_ord = norm_ord
        self._tol = tol
        self._key = jax.random.PRNGKey(random_state)

    def fit(self, X):
        self.codebook, _ = _kmeans_run(self._key, X, self.n_clusters,
                                      self._norm_ord, thresh=self._tol)
        return self

    def fit_predict(self, x):
        return self.fit(x).predict(x)

    def predict(self, X):
        assignment, _ = _vector_quantize(X, self.codebook, 2)
        return assignment


class KMeansPlusPlus(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters=8, num_seeds=10, norm_ord=2,
                 tol=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self.num_seeds = num_seeds
        self._norm_ord = norm_ord
        self._tol = tol
        self._key = jax.random.PRNGKey(random_state)

    def fit(self, X):
        self.codebook, _ = _kmeans(self._key, X, self.n_clusters,
                                  self._norm_ord, self.num_seeds,
                                  thresh=self._tol)
        return self

    def fit_predit(self, x):
        return self.fit(x).predict(x)

    def predict(self, X):
        assignment, _ = _vector_quantize(X, self.codebook, 2)
        return assignment
