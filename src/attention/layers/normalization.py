import jax.numpy as jnp
import haiku as hk
import jax


EPSILON = 1e-5

class WIPLayerNorm(hk.Module):
    """TODO: Add scaling and bias"""
    def __init__(self, name=None):
        super().__init__(name=name)


    def __call__(self, x: jnp.ndarray):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        eps = jax.lax.convert_element_type(EPSILON, variance.dtype)
        inv_variance = jax.lax.rsqrt(variance + eps)
        return inv_variance * (x - mean)
        

