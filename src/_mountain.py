from functools import partial
import jax
import jax.numpy as jnp
from jax.ops import index, index_update
import numpy as onp
from sklearn.base import ClusterMixin, BaseEstimator

@partial(jax.jit, static_argnums=(1, 2))
def initialize(X, gran=5, precision=16):
    dtype = jnp.float16 if precision == 16 else jnp.float32
    
    X = X.astype(dtype)

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    dims = [jnp.linspace(mi, mx, gran, dtype=dtype)
            for (mi, mx) in zip(mins, maxs)]

    prototypes = jnp.stack(jnp.meshgrid(*dims))

    num_dims = len(dims)

    prototypes = prototypes[jnp.newaxis, ...]
    X = jnp.expand_dims(X, axis=tuple(range(2, num_dims + 2)))

    return prototypes, X

@partial(jax.jit, static_argnums=(2,))
def mountain_function(prototypes, x, sigma):
    points_norm = jnp.linalg.norm(x, ord=2, axis=1)
    grid_norm = jnp.linalg.norm(prototypes, ord=2, axis=1)

    cosine = (x * prototypes).sum(axis=1) / (points_norm * grid_norm)

    return jnp.exp(cosine / (2 * sigma ** 2)).sum(axis=0)

@jax.jit
def get_cluster(prototypes, prototypes_density):
    num_dims = prototypes.shape[1]
    cluster_id = jnp.unravel_index(
        jnp.argmax(prototypes_density),
        prototypes_density.shape
    )
    cluster = prototypes[(0, tuple(range(num_dims)), *cluster_id)]
    cluster = jnp.expand_dims(
        cluster, axis=tuple(range(1, num_dims + 1))
    )[jnp.newaxis, ...]
    return cluster, prototypes_density[cluster_id]

@partial(jax.jit, static_argnums=(0, 1))
def stop(thresh, initial_state, state):
    return (state[-1] / initial_state[-1]) > thresh

@partial(jax.jit, static_argnums=(0,))
def mountain_update(params, val):
    prototypes, clusters, X, idx, prototypes_density, cluster_density = val
    sigma, beta = params
    num_dims = prototypes.shape[1]

    cluster = jnp.expand_dims(
        clusters[idx],
        tuple(range(1, num_dims + 1))
    )[jnp.newaxis, ...]

    idx += 1

    prototype_norm = jnp.linalg.norm(prototypes[0], ord=2, axis=0)
    cluster_norm = jnp.linalg.norm(cluster[0], ord=2, axis=0)

    cluster_mass = jnp.exp(
        -jnp.sum(
            prototypes[0] * cluster[0], axis=0
        ) / (cluster_norm * prototype_norm) / (2 * beta ** 2)
    )

    near_cluster_density = mountain_function(cluster, X, sigma) * cluster_mass

    new_prototypes_density = prototypes_density - near_cluster_density
    new_cluster, cluster_density = get_cluster(prototypes, new_prototypes_density)
    clusters = index_update(clusters, index[idx], jnp.squeeze(new_cluster))

    val = (prototypes, clusters, X, idx, new_prototypes_density, cluster_density)

    return val

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def mountain_run(X, sigma, beta, gran, thresh, precision):
    prototypes, X = initialize(X, gran, precision)
    prototypes_density = mountain_function(prototypes, X, sigma)
    cluster, cluster_density = get_cluster(prototypes, prototypes_density)

    idx = 0
    params = (sigma, beta)

    clusters = jnp.squeeze(jnp.zeros_like(X)) + jnp.nan

    clusters = index_update(
        clusters,
        index[idx],
        jnp.squeeze(cluster)
    )

    val = (prototypes, clusters, X, idx, prototypes_density, cluster_density)

    state = jax.lax.while_loop(
        partial(stop, thresh, val),
        partial(mountain_update, params),
        val
    )

    return state


class Mountain(ClusterMixin, BaseEstimator):
    def __init__(self, sigma=0.1, beta=0.1, precision=16,
                 gran=5, tol=0.01, random_state=42):

        assert precision in {16, 32}, 'wrong precision'

        self.sigma = sigma
        self.beta = beta
        self.gran = gran
        self.tol = tol
        self.precision = precision
        self._key = jax.random.PRNGKey(random_state)

    def fit(self, X):
        prototypes, clusters, _, idx, prototypes_density, cluster_density = mountain_run(
            X, self.sigma, self.beta, self.gran, self.tol, self.precision
        )
        self.prototypes = prototypes
        self.clusters = clusters[:idx+1]
        self.prototypes_density = prototypes_density
        self.cluster_density = cluster_density
        return self

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def predict(self, X):
        clusters_norm = jnp.linalg.norm(
            moun.clusters, ord=2, axis=1
        )[:, jnp.newaxis]
        points_norm = jnp.linalg.norm(X, ord=2, axis=1)

        assignment = jnp.argmin(
            (moun.clusters @ X.T) / (clusters_norm * points_norm),
            axis=0
        )

        return assignment
