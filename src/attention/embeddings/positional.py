import jax.numpy as jnp


def sin_embeddings(pos: jnp.ndarray, dim: int) -> jnp.ndarray:
    pos = jnp.expand_dims(pos, -1)
    wavelength = jnp.power(10_000, 2 * jnp.arange(dim) / dim)
    return jnp.sin(pos / wavelength)


def cos_embeddings(pos: jnp.ndarray, dim: int) -> jnp.ndarray:
    pos = jnp.expand_dims(pos, -1)
    wavelength = jnp.power(10_000, 2 * jnp.arange(dim) / dim)
    return jnp.cos(pos / wavelength)
