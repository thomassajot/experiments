from jax import lax
from jax import numpy as jnp


def softmax(x, axis=-1):
    x_max = jnp.max(x, axis=axis, keepdims=True)
    exp_x = jnp.exp(x - lax.stop_gradient(x_max))
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)


def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return jnp.maximum(0, x)
