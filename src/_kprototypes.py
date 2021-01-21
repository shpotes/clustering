from functools import partial
import jax
import jax.numpy as jnp
from sklearn.base import ClusterMixin, BaseEstimator

@partial(jax.jit, static_argnums=(4, 5))
def _compute_assignments(
    numerical_points,
    categorical_points,
    numerical_prototypes,
    categorical_prototypes,
    norm_ord,
    gamma
):
    norm = partial(jnp.linalg.norm, ord=norm_ord)

    numerical_dist = jax.vmap(
        lambda point: jax.vmap(norm)(numerical_prototypes - point)
    )(numerical_points)
    categorical_dist = jax.vmap(
        lambda point: jax.vmap(jnp.sum)(categorical_prototypes != point)
    )(categorical_points)

    dist = numerical_dist + gamma * categorical_dist

    assignment = jnp.argmin(dist, axis=1)

    numerical_cost = jax.vmap(norm)(
        numerical_prototypes[assignment, :] - numerical_points
    ).sum()
    categorical_cost = jax.vmap(jnp.sum)(
        categorical_prototypes[assignment, :] == categorical_points
    ).sum()

    cost = numerical_cost + gamma * categorical_cost # Review paper gamma

    return assignment, cost

@partial(jax.jit, static_argnums=(2, 3))
def improve_centroids(
        numerical_points,
        categorical_points,
        norm_ord,
        gamma,
        k,
        state,
):
    (prev_numerical_proto, prev_categorical_proto), prev_assignemnt, _, _ = state

    assignment, cost = _compute_assignments(
        numerical_points,
        categorical_points,
        prev_numerical_proto,
        prev_categorical_proto,
        norm_ord,
        gamma
    )

    new_numerical_proto = jnp.mean(
        jnp.where(
            assignment.reshape(-1, 1, 1) == jnp.arange(k).reshape(1, k, 1),
            numerical_points[:, jnp.newaxis, :],
            0
        ),
        axis=0
    )

    new_categorical_proto = jnp.apply_along_axis(
        lambda x: jnp.bincount(x, length=categorical_points.max() + 1).argmax(),
        axis=0,
        arr=jnp.where(
            assignment.reshape(-1, 1, 1) == jnp.arange(k).reshape(1, k, 1),
            categorical_points[:, jnp.newaxis, :],
            categorical_points.max() + 10
        )
    )

    return (
        (new_numerical_proto, new_categorical_proto),
        assignment, prev_assignemnt, cost
    )

@partial(jax.jit, static_argnums=(3, 4, 5))
def _kprototypes_run(key, numerical_points, categorical_points, k, gamma, norm_ord):
    num_points = numerical_points.shape[0]

    initial_indices = jax.random.permutation(key, jnp.arange(num_points))[:k]
    initial_numerical_centroids = numerical_points[initial_indices, :]
    initial_categorical_centroids = categorical_points[initial_indices, :]
    # state = (centroids, assignment, prev_assignemnt)
    initial_state = (
        (initial_numerical_centroids, initial_categorical_centroids),
        jnp.zeros(num_points, dtype=jnp.int32) - 1,
        jnp.ones(num_points, dtype=jnp.int32),
        jnp.inf
    )

    update_function = partial(
        improve_centroids,
        numerical_points,
        categorical_points,
        norm_ord,
        gamma,
        k
    )

    centroids, assignment, _, cost = jax.lax.while_loop(
        lambda state: jnp.all(state[1] != state[2]),
        update_function,
        initial_state
    )

    return centroids, assignment, cost

@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def _kprototypes(key, numerical_points, categorical_points, restarts, k, gamma, norm_ord):
    all_centroids, all_assignment, all_distortions = jax.vmap(
        lambda key: _kprototypes_run(key, numerical_points, categorical_points, k, gamma, norm_ord)
    )(jax.random.split(key, restarts))
    i = jnp.argmin(all_distortions)

    all_numerical_centroids, all_categorical_centroids = all_centroids

    return (
        (all_numerical_centroids[i], all_categorical_centroids[i]),
        all_assignment[i],
        all_distortions[i]
    )


class KPrototypes(ClusterMixin, BaseEstimator):
    def __init__(
            self,
            n_clusters,
            norm_ord=2,
            n_seeds=100,
            random_state=42
    ):

        self.n_clusters = n_clusters
        self.gamma = None
        self.norm_ord = norm_ord
        self.n_seeds = n_seeds
        self._key = jax.random.PRNGKey(random_state)

        self.numerical_centroids = None
        self.categorical_centroids = None

    def _compute_gamma(self, numerical_points):
        return jnp.std(numerical_points, axis=0).mean()

    def fit(self, points, categorical_mask):
        numerical_points = points[:, ~categorical_mask]
        categorical_points = points[:, categorical_mask]

        if self.gamma is None:
            self.gamma = self._compute_gamma(numerical_points)

        centroids, assignment, cost = _kprototypes(
            self._key,
            numerical_points,
            categorical_points,
            self.n_seeds,
            self.n_clusters,
            self.gamma,
            self.norm_ord
        )

        self.numerical_prototypes, self.categorical_prototypes = centroids
        return self

    def predict(self, points, categorical_mask):
        numerical_points = points[:, ~categorical_mask]
        categorical_points = points[:, categorical_mask]

        assignment, cost = _compute_assignments(
            numerical_points,
            categorical_points,
            self.numerical_prototypes,
            self.categorical_prototypes,
            self.norm_ord,
            self.gamma
        )

        return assignment

    def fit_predict(self, points, categorical_mask):
        return self.fit(points, categorical_mask).predict(points, categorical_mask)
