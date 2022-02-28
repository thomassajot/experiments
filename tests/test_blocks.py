from attention.layers.encoder import EncoderBlock
import haiku as hk
import jax.numpy as jnp
import jax

def test_encoder_block():
    def encoder(x):
        encoder = EncoderBlock(num_heads=2, key_size=16, value_size=32, model_size=8, name='blob')
        return encoder(x)

    encoder = hk.transform(encoder)
    rng = jax.random.PRNGKey(42)
    x = jnp.ones([8, 224, 224, 8])
    params = encoder.init(rng, x)
    encoded = encoder.apply(params, rng, x)

    assert encoded.shape == (8, 224, 224, 8)