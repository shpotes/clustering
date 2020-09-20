import jax.numpy as jnp
import jax.numpy.linalg as LA

def cosine_similarity(a, b, axis=None):
    normalized_a = a / LA.norm(a, ord=2, axis=axis).reshape(-1, 1)
    normalized_b = b / LA.norm(b, ord=2, axis=axis).reshape(-1, 1)

    return (normalized_a @ normalized_b.T)
