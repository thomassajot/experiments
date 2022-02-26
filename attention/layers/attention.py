from jax import numpy as jnp
from attention.activations import softmax


def scaled_dot_product_attention(query, key, value):
    d_k = query.shape[1]
    d = jnp.dot(query, key.T)
    attentions = softmax(d / jnp.sqrt(d_k))
    return jnp.dot(attentions, value)



def multi_head_attention(params, q, k, v):
    *head_params, concat_w = params
    activations = []
    for q_w, k_w, v_w in head_params:
        head_activation = scaled_dot_product_attention(jnp.dot(q, q_w), jnp.dot(k, k_w), jnp.dot(v, v_w))
        activations.append(head_activation)
    return jnp.dot(jnp.concatenate(activations), concat_w)
