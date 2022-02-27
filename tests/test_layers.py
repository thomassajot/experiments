from jax import numpy as jnp
import pytest
from attention.layers.attention import MultiHeadAttention
import numpy as np
import haiku as hk
import jax

def test_multi_head_attention():
    num_heads = 2
    sequence = 3
    batch = 1
    d_k = 4
    d_v = 2
    output_size = 5
    rng = jax.random.PRNGKey(42)

    shape = (batch, sequence, d_k)
    query = jnp.arange(np.prod(shape), dtype=float).reshape(shape)
    key = jnp.arange(np.prod(shape), dtype=float).reshape(shape)
    shape = (batch, sequence, d_v)
    value = jnp.arange(np.prod(shape), dtype=float).reshape(shape)

    
    def multi_head_attention(query, key, value):
        mha = MultiHeadAttention(num_heads=num_heads, key_size=d_k, value_size=d_v, model_size=output_size, name='mha')
        return mha(query, key, value)

    multi_head_attention = hk.transform(multi_head_attention)
    rng, subrng = jax.random.split(rng)
    params = multi_head_attention.init(subrng, query, key, value)
    activations = multi_head_attention.apply(params, rng, query, key, value)

    expected_activations = np.array([[
        [ 7.312496 ,  0.468923 , -4.265187 ,  2.7021663, -6.186495 ],
        [ 8.041342 ,  0.4608997, -4.6220317,  2.9440174, -6.811014 ],
        [ 8.050848 ,  0.460734 , -4.626566 ,  2.9473486, -6.819031 ]
    ]])

    assert activations.shape == (batch, sequence, output_size)
    np.testing.assert_almost_equal(np.array(activations), expected_activations, decimal=6)