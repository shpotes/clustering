from functools import partial
import jax
import jax.numpy as jnp
import jax.numpy.linalg as LA

def cosine_similarity(a, b, norm_axis=1):
    normalized_a = a / LA.norm(a, ord=2, axis=norm_axis).reshape(-1, 1)
    normalized_b = b / LA.norm(b, ord=2, axis=norm_axis).reshape(-1, 1)

    return (normalized_a @ normalized_b.T)

def compute_kmeans_distance(points, prototypes, norm_ord):
    if norm_ord == 'cosine':
        return 1 - cosine_similarity(points, prototypes, norm_axis=1)

    norm = partial(LA.norm, ord=norm_ord)

    dist = jax.vmap(
        lambda point: jax.vmap(norm)(prototypes - point)
    )(points)

    return dist

def compute_kmeans_cost(points, prototypes, assignment, norm_ord):
    if norm_ord == 'cosine':
        dist_matrix = 1 - cosine_similarity(points, prototypes, norm_axis=1)
        return dist_matrix[jnp.arange(len(assignment)), assignment].mean()

    norm = partial(LA.norm, ord=norm_ord)

    cost = jax.vmap(norm)(
        prototypes[assignment, :] - points
    ).sum()

    return cost
