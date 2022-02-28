import haiku as hk
import jax.numpy as jnp
from jax.nn import relu

class MLP(hk.Module):
    def __init__(self, sizes, name=None):
        super().__init__(name=None)
        self.sizes = sizes

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs 
        
        for output_size in self.sizes[:-1]:
            x = hk.Linear(output_size=output_size)(x)
            x = relu(x)

        return hk.Linear(output_size=self.sizes[-1])(x)