from jax import numpy as jnp
import pytest
from attention.layers.attention import scaled_dot_product_attention
import numpy as np


@pytest.mark.parametrize('query, key, value, expected', [
    ([[1, 0]], [[1, 0]], [[1]], [[1]]),  # key and queries have the same shape but not value necessarily
    ([[1, 0]], [[1e5, 0], [0, 0]], [[1, 1], [2, 2]], [[1, 1]]),
    ([[1, 1]], [[1, 0], [1, 0]], [[1, 1], [2, 2]], [[1.5, 1.5]]),
    # use input produced with the inverse function of the attention weights
    ([[1, 1]], [[np.sqrt(2) * np.log(9), 0], [np.sqrt(2) * np.log(1), 0]], [[1, 0], [2, 0]], [[0.9 + 0.2, 0]]),
    ([[1, 1]], [[1, 1], [2, 2]], [[1, 3], [2, 4]], [[1.8044298, 3.8044298]]),
])
def test_scaled_dot_product_attention(query, key, value, expected):
    query = jnp.array(query)
    key = jnp.array(key)
    value = jnp.array(value)
    activation = scaled_dot_product_attention(query, key, value)
    
    np.testing.assert_almost_equal(np.array(activation), np.array(expected))
