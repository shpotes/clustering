from functools import partial
import jax
import jax.numpy as jnp
from jax.ops import index, index_update
import numpy as onp
from sklearn.base import ClusterMixin, BaseEstimator

@partial(jax.jit, static_argnums=(2, 3,))
def mountain_function(x, d, norm_ord, r_a):
    norm = partial(jnp.linalg.norm, ord=norm_ord)
    return jnp.exp(-norm((d - x), axis=1) / ((r_a / 2) ** 2)).sum(axis=-1)

@jax.jit
def get_cluster(x, density):
    cluster_idx = jnp.argmax(density)
    cluster = x[cluster_idx]
    return cluster, density[cluster_idx]

@partial(jax.jit, static_argnums=(0,))
def subtractive_update(params, val):
    idx, x, density, clusters, cluster_density = val
    norm_ord, r_a = params
    r_b = 1.5 * r_a

    cluster = clusters[idx]
    idx += 1

    cluster_mass = jnp.linalg.norm(
        x - cluster.reshape(1, -1, 1),
        ord=norm_ord, axis=1
    ).sum()

    near_cluster_density = cluster_mass * jnp.exp(
        -jnp.linalg.norm(
            x-  cluster.reshape(1, -1, 1),
            ord=norm_ord, axis=1
        )
    )

    new_density = density - jnp.squeeze(near_cluster_density)
    
    new_cluster, cluster_density = get_cluster(x, new_density)

    clusters = index_update(clusters, index[idx], jnp.squeeze(new_cluster))

    val = (idx, x, density, clusters, cluster_density)

    return val

@partial(jax.jit, static_argnums=(0, 1))
def stop(thresh, initial_state, state):
    return (state[-1] / initial_state[-1]) > thresh

@partial(jax.jit, static_argnums=(1, 2, 3))
def subtractive_run(x, norm_ord, r_a, thresh):

    d = x.T[jnp.newaxis, ...]
    x = x[..., jnp.newaxis]

    density = mountain_function(x, d, norm_ord, r_a)
    cluster, cluster_density = get_cluster(x, density)

    idx = 0
    params = (norm_ord, r_a)

    clusters = jnp.squeeze(jnp.zeros_like(x)) + jnp.nan

    clusters = index_update(
        clusters,
        index[idx],
        jnp.squeeze(cluster)
    )

    val = (idx, x, density, clusters, cluster_density)

    state = jax.lax.while_loop(
        partial(stop, thresh, val),
        partial(subtractive_update, params),
        val
    )

    return state


class Substractive(ClusterMixin, BaseEstimator):
    def __init__(self, norm_ord=2, r_a=0.1, precision=16,
                 tol=0.1, random_state=42):

        assert precision in {16, 32}, 'wrong precision'

        self.norm_ord = norm_ord
        self.r_a = r_a
        self.tol = tol

        self._key = jax.random.PRNGKey(random_state)
        self._dtype = jnp.float16 if precision == 16 else jnp.float32

    def fit(self, X):
        x = jnp.array(X, dtype=self._dtype)

        idx, x, density, clusters, cluster_density  = subtractive_run(
            x, self.norm_ord, self.r_a, self.tol
        )

        self.density = density
        self.clusters = clusters[:idx+1]
        self.cluster_density = cluster_density

        return self

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def predict(self, X):
        X = jnp.array(X, dtype=self._dtype)

        norm = partial(jnp.linalg.norm, ord=self.norm_ord)
        assignment = jax.vmap(
            lambda points: jnp.argmin(jax.vmap(norm)(self.clusters - points))
        )(X)
        self.labels_ = onp.array(assignment)
        return assignment
