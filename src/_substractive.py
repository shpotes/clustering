from functools import partial
import jax
import jax.numpy as jnp
from jax.ops import index, index_update
import numpy as onp
from sklearn.base import ClusterMixin, BaseEstimator
from .utils import cosine_similarity

@partial(jax.jit, static_argnums=(2,))
def density_function(points, prototypes, r_a):
    """
    Compute the density meassure for the substractive clustering (Moertini, 2002)

    Parameters
    ----------
    points : float[N, M]
        Points to be clusters (N points, M dimensions)
    prototypes : float[Np, M]
        Clusters prototypes (N prototypes, M dimensions)
    r_a : float
        neighborhood radius

    Returns
    -------
    type 
        float[Np]
    """
    density = exp_dist(points, prototypes, r_a)
    return jnp.sum(density, axis=0)

@partial(jax.jit, static_argnums=(2,))
def exp_dist(points, prototypes, radius):
    """
    Compute the density arround each prototype

    Parameters
    ----------
    points : float[N, M]
        Points to be clusters (N points, M dimensions)
    prototypes : float[Np, M]
        Clusters prototypes (N prototypes, M dimensions)
    r_a : float
        neighborhood radius

    Returns
    -------
    type 
        float[N, Np]
    """

    cosine_dist = 1 - cosine_similarity(points, prototypes, norm_axis=1)
    return jnp.exp(-cosine_dist / (radius / 2)**2)

@jax.jit
def get_cluster(x, density):
    """
    Return the next cluster

    Parameters
    ----------
    x : float[Np, M]
        Cluster candidates (Np points, M dimensions)
    density : float[Np]
        The precomputed density

    Returns
    -------
    cluster: float[M]
        The cluster that maximize the density
    density : float
        Their corresponding density
    """

    cluster_idx = jnp.argmax(density)
    cluster = x[cluster_idx]
    return cluster, density[cluster_idx]

@partial(jax.jit, static_argnums=(0,))
def subtractive_update(r_a, val):
    idx, x, density, clusters, cluster_density = val
    r_b = 1.5 * r_a

    cluster = clusters[idx][jnp.newaxis, :]
    idx += 1

    cluster_mass = density_function(x, cluster, r_a)
    near_cluster_density = cluster_mass * exp_dist(x, cluster, r_b)

    new_density = density - jnp.squeeze(near_cluster_density)
    
    new_cluster, cluster_density = get_cluster(x, new_density)

    clusters = index_update(clusters, index[idx], jnp.squeeze(new_cluster))

    val = (idx, x, density, clusters, cluster_density)

    return val

@partial(jax.jit, static_argnums=(0, 1))
def stop(thresh, initial_state, state):
    return (state[-1] / initial_state[-1]) > thresh

@partial(jax.jit, static_argnums=(1, 2))
def subtractive_run(x, r_a, thresh):

    density = density_function(x, x, r_a)
    cluster, cluster_density = get_cluster(x, density)

    idx = 0

    # Hacky dynamic storage in jax
    clusters = jnp.squeeze(jnp.zeros_like(x)) + jnp.nan

    clusters = index_update(
        clusters,
        index[idx],
        jnp.squeeze(cluster)
    )

    initial_state = (idx, x, density, clusters, cluster_density)

    state = jax.lax.while_loop(
        partial(stop, thresh, initial_state),
        partial(subtractive_update, r_a),
        initial_state
    )

    return state


class Substractive(ClusterMixin, BaseEstimator):
    def __init__(self, r_a=2, precision=16,
                 tol=0.1, random_state=42):

        assert precision in {16, 32}, 'wrong precision'
        self.r_a = r_a
        self.tol = tol

        self._key = jax.random.PRNGKey(random_state)
        self._dtype = jnp.float16 if precision == 16 else jnp.float32

    def fit(self, X):
        x = jnp.array(X, dtype=self._dtype)

        idx, x, density, clusters, cluster_density  = subtractive_run(
            x, self.r_a, self.tol
        )

        self.n_clusters = idx + 1
        self.density = density
        self.clusters = clusters[:self.n_clusters]
        self.cluster_density = cluster_density

        return self

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def predict(self, X):
        X = jnp.array(X, dtype=self._dtype)

        dist_matrix = 1 - cosine_similarity(X, self.clusters, norm_axis=1)
        assignment = jnp.argmin(dist_matrix, axis=1)

        self.labels_ = onp.array(assignment)
        return assignment
