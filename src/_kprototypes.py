from functools import partial
import jax
from jax.ops import index, index_update
import jax.numpy as jnp
import numpy as onp
from sklearn.base import ClusterMixin, BaseEstimator
from .utils import compute_kmeans_distance, compute_kmeans_cost

FILL_VAL = -2147483648

@partial(jax.jit, static_argnums=(4, 5, 6))
def _compute_assignments(
        numerical_points,
        categorical_points,
        numerical_prototypes,
        categorical_prototypes,
        norm_ord,
        gamma,
        nan_friendly
):
    if nan_friendly:
        numerical_points, categorical_points = _fill_nans(
            numerical_points,
            numerical_prototypes,
            categorical_points,
            categorical_prototypes
        )

    numerical_dist = compute_kmeans_distance(
        numerical_points, numerical_prototypes, norm_ord
    )
    categorical_dist = jax.vmap(
        lambda point: jax.vmap(jnp.sum)(categorical_prototypes != point)
    )(categorical_points)

    dist = numerical_dist + gamma * categorical_dist

    assignment = jnp.argmin(dist, axis=1)

    numerical_cost = compute_kmeans_cost(
        numerical_points, numerical_prototypes, assignment, norm_ord
    )

    categorical_cost = jax.vmap(jnp.sum)(
        categorical_prototypes[assignment, :] == categorical_points
    ).sum()

    cost = numerical_cost + gamma * categorical_cost # Review paper gamma

    return assignment, cost

def mode(arr, axis=0, max_value=250):
    return jnp.apply_along_axis(
        lambda x: jnp.bincount(x, length=max_value).argmax(),
        axis=axis,
        arr=arr.astype(jnp.int32)
    )

@partial(jax.jit, static_argnums=(2,))
def _update_categorical_cluster(categorical_points, assignment, k):
    catdidates = jnp.where(
        assignment.reshape(-1, 1, 1) == jnp.arange(k).reshape(1, k, 1),
        categorical_points[:, jnp.newaxis, :],
        255 # arbitrary uper bound
    )

    new_clusters = mode(catdidates)

    return new_clusters

def _update_numerical_cluster(numerical_points, assignment, k):
    return jnp.mean(
        jnp.where(
            assignment.reshape(-1, 1, 1) == jnp.arange(k).reshape(1, k, 1),
            numerical_points[:, jnp.newaxis, :],
            0
        ),
        axis=0
    )

def _initialize_centroids(key, numerical_points, categorical_points, k, fill_nan=False):
    n_points = numerical_points.shape[0]

    initial_indices = jax.random.permutation(key, jnp.arange(n_points))[:k]
    numerical_cluster = numerical_points[initial_indices]

    initial_indices = jax.random.permutation(key, jnp.arange(n_points))[:k]
    categorical_cluster = categorical_points[initial_indices]

    if fill_nan:
        numerical_cluster, categorical_cluster = _fill_nans(
            numerical_clusters,
            numerical_points,
            categorical_clusters,
            categorical_points
        )
    
    return numerical_cluster, categorical_cluster

def _fill_nans(numerical_points, numerical_clusters, categorical_points, categorical_clusters):
    filled_num_nan = jnp.isnan(numerical_points) * jnp.nanmean(numerical_clusters, axis=0)
    numerical_points = jnp.nan_to_num(numerical_points) + filled_num_nan

    filled_cat_nan = (categorical_points == FILL_VAL) * mode(categorical_clusters, axis=0)
    categorical_points = jnp.where(
        categorical_points == FILL_VAL,
        filled_cat_nan,
        categorical_points,
    )

    return numerical_points, categorical_points

@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def _kprototypes_run(key, numerical_points, categorical_points, k, gamma, norm_ord, nan_friendly):
    def improve_centroids(k, gamma, norm_ord, state):
        (prev_numerical_proto, prev_categorical_proto), prev_assignemnt, _, _ = state        
                
        assignment, cost = _compute_assignments(
            numerical_points,
            categorical_points,
            prev_numerical_proto,
            prev_categorical_proto,
            norm_ord,
            gamma,
            nan_friendly
        )

        new_numerical_proto = _update_numerical_cluster(numerical_points, assignment, k)
        new_categorical_proto = _update_categorical_cluster(categorical_points, assignment, k)

        return (
            (new_numerical_proto, new_categorical_proto),
            assignment, prev_assignemnt, cost
        )

    n_points = numerical_points.shape[0]

    initial_numerical_centroids, initial_categorical_centroids = _initialize_centroids(
        key, numerical_points, categorical_points, k
    )

    # state :: (centroids, assignment, prev_assignemnt)
    initial_state = (
        (initial_numerical_centroids, initial_categorical_centroids),
        jnp.zeros(n_points, dtype=jnp.int32) - 1,
        jnp.ones(n_points, dtype=jnp.int32),
        jnp.inf
    )

    update_function = partial(
        improve_centroids,
        k, gamma, norm_ord
    )

    centroids, assignment, _, cost = jax.lax.while_loop(
        lambda state: jnp.all(state[1] != state[2]),
        update_function,
        initial_state
    )

    return centroids, assignment, cost

@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def _kprototypes(key, numerical_points, categorical_points, restarts, k, gamma, norm_ord, nan_friendly):
    all_centroids, all_assignment, all_distortions = jax.vmap(
        lambda key: _kprototypes_run(
            key,
            numerical_points,
            categorical_points.astype(jnp.int32),
            k, gamma, norm_ord, nan_friendly
        )
    )(jax.random.split(key, restarts))
    i = jnp.argmin(all_distortions)

    all_numerical_centroids, all_categorical_centroids = all_centroids

    return (
        (all_numerical_centroids[i], all_categorical_centroids[i]),
        all_assignment[i],
        all_distortions[i]
    )


class KPrototypes:
    def __init__(
            self,
            n_clusters,
            norm_ord=2,
            gamma=None,
            nan_friendly=False,
            n_seeds=100,
            random_state=42
    ):

        self.n_clusters = n_clusters
        self.gamma = gamma
        self.norm_ord = norm_ord
        self.n_seeds = n_seeds
        self.nan_friendly = nan_friendly
        self._key = jax.random.PRNGKey(random_state)

        self.centroids = None

    def _compute_gamma(self, numerical_points):
        return onp.std(numerical_points, axis=0).mean()

    def fit(self, points, categorical_mask):
        numerical_points = points[:, ~categorical_mask]
        categorical_points = points[:, categorical_mask]

        if self.gamma is None:
            self.gamma = self._compute_gamma(numerical_points)

        centroids, _, cost = _kprototypes(
            self._key,
            numerical_points,
            categorical_points,
            self.n_seeds,
            self.n_clusters,
            self.gamma,
            self.norm_ord,
            self.nan_friendly
        )

        numerical_prototypes, categorical_prototypes = centroids
        self.centroids = onp.zeros((self.n_clusters, len(categorical_mask)))
        self.centroids[:, categorical_mask] = onp.array(categorical_prototypes)
        self.centroids[:, ~categorical_mask] = onp.array(numerical_prototypes)

        return self

    def predict(self, points, categorical_mask):
        numerical_points = points[:, ~categorical_mask]
        categorical_points = points[:, categorical_mask]

        assignment, cost = _compute_assignments(
            numerical_points,
            categorical_points,
            self.centroids[:, ~categorical_mask],
            self.centroids[:, categorical_mask],
            self.norm_ord,
            self.gamma,
            self.nan_friendly
        )

        return assignment

    def fit_predict(self, points, categorical_mask):
        return self.fit(points, categorical_mask).predict(points, categorical_mask)
