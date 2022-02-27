from jax import numpy as jnp
import haiku as hk
import numpy as np 
import jax


class MultiHeadAttention(hk.Module):

    def __init__(self, num_heads: int, key_size: int, value_size: int, model_size: int , name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size
        self.model_size = model_size
        self.w_init = hk.initializers.VarianceScaling()

    def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray):
        # Query shape: [..., Sequence (s), d]
       query_heads = self._linear_layer(query, self.key_size, "query")
       key_heads = self._linear_layer(key, self.key_size, "key")
       value_heads = self._linear_layer(value, self.value_size, "value")

        # for each head, we have for each element in a sequence the attention weights for all element in the sequence
        # [sequence size, num_heads, dimension] => [num_heads, sequence, sequence]
       attention_logits = jnp.einsum('...shd,...Shd->...hsS', query_heads, key_heads)
       # scaled 
       sqrt_key_size = np.sqrt(self.key_size)
       attention_weights = jax.nn.softmax(attention_logits / sqrt_key_size)
       attention = jnp.einsum('...hsS,...Shd->...shd', attention_weights, value_heads)
       # we get the same shape as the input except for the last dimention which is the concatenation along the heads
       attention = attention.reshape((*query.shape[:-1], -1))
       return hk.Linear(self.model_size, w_init=self.w_init)(attention)

    
    def _linear_layer(self, x: jnp.ndarray, head_size: int, name=None) -> jnp.ndarray:
        """Apply a linear layer to query, key, values for all heads
        x.shape = [Batch, Sequence, head_size] -> [Batch, Sequence, Heads, head_size]
        """
        y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
        return y.reshape((*x.shape[:-1], self.num_heads, head_size))