from jax import numpy as jnp
import haiku as hk
import numpy as np 
import jax

from attention.layers.attention import MultiHeadAttention
from attention.layers.mlp import MLP
from attention.layers.normalization import WIPLayerNorm


class EncoderBlock(hk.Module):

    def __init__(self, num_heads: int, key_size: int, value_size: int, model_size: int, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size
        self.model_size = model_size

    def __call__(self, x: jnp.ndarray):
        mha = MultiHeadAttention(num_heads=self.num_heads, 
                                 key_size=self.key_size, 
                                 value_size=self.value_size, 
                                 model_size=self.model_size)
        wip_layer_norm_attn = WIPLayerNorm()
        wip_layer_norm_linear = WIPLayerNorm()

        attention_activations = mha(query=x, key=x, value=x)
        residual = attention_activations + x
        x = wip_layer_norm_attn(residual)

        feed_forward_x = MLP(sizes=[self.model_size, self.model_size], name='linear')(x)
        residual = feed_forward_x + x
        return wip_layer_norm_linear(residual)