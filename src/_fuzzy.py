from functools import partial
import jax
import jax.numpy as jnp
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

@partial(jax.jit, static_argnums=(2,))
def get_cluster(points, membership, m):
    return jnp.sum(
        membership[..., jnp.newaxis] ** m * points, axis=1
    ) / jnp.sum(membership, axis=1)[0]

@partial(jax.jit, static_argnums=(3,))
def update_membership(points, cluster, membership, m):
    c, num_dims = cluster.shape
    
    dist = jnp.linalg.norm(
        (cluster.reshape(c, 1, num_dims) - points),
        ord=norm_ord, axis=2
    )

    new_cost = jnp.sum(membership ** m * dist ** 2)

    tmp_dist = dist ** (2 / (m - 1))
    tmp_dist /= tmp_dist.sum(axis=0)
    new_membership = 1 / tmp_dist
    new_membership /= new_membership.sum(axis=0)

    return new_membership


@partial(jax.jit, static_argnums=(0,))
def improve_membership(params, val):
    membership, points, _, old_cost = val
    norm_ord, m = params

    cluster = get_cluster(points, membership, m)
    new_membership = update_membership(points, cluster, membership, m)

    return new_membership, points, old_cost, new_cost

@partial(jax.jit, static_argnums=(2, 4, 3))
def cmeans(key, points, c, norm_ord=2, tol=1e-5):
    num_samples, num_dims = points.shape

    membership = jax.random.uniform(key, (c, num_samples))
    membership /= membership.sum(axis=0)

    initial_val = (membership, points, jnp.inf, 0.)
    params = (norm_ord, m)

    state = jax.lax.while_loop(
        lambda val: abs(val[-1] - val[-2]) > tol,
        partial(improve_membership, params),
        initial_val
    )

    membership, points, other, cost = state
    cluster = get_cluster(points, membership, m)

    return cluster, membership, points, cost, other

class CMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters=8, norm_ord=2, tol=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self._norm_ord = norm_ord
        self._tol = tol
        self._key = jax.random.PRNGKey(random_state)

    def fit(self, X):
        state = cmeans(self._key, X, self.n_clusters,
                       self._norm_ord, self._tol)

        self.cluster, self.membership, _, self.cost = state

        return self

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def predict(self, X):
        return self.cluster, update_membership(X, self.cluster, self.membership, m)
        
