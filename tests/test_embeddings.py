import pytest
import jax.numpy as jnp
from attention.embeddings.positional import sin_embeddings, cos_embeddings
import numpy as np


@pytest.mark.parametrize(
    "positions, dim, expected_embeddings",
    [
        (jnp.array([0, 1]), 1, np.array([0, 0.8414]).reshape(2, 1)),
        (
            jnp.ones((1, 1, 2)),
            4,
            np.array([8.4147e-01, 9.9998e-03, 9.9999e-05, 1.0e-06] * 2).reshape(1, 1, 2, 4),
        ),
    ],
)
def test_positional_sin_embeddings(positions, dim, expected_embeddings):
    embeddings = sin_embeddings(pos=positions, dim=dim)
    assert embeddings.shape == expected_embeddings.shape
    np.testing.assert_allclose(np.array(embeddings), expected_embeddings, rtol=1e-4, atol=0)


@pytest.mark.parametrize(
    "positions, dim, expected_embeddings",
    [
        (jnp.array([0, 1]), 1, np.array([1, 0.5403]).reshape(2, 1)),
        (
            jnp.ones((1, 1, 2)),
            4,
            np.array([0.540302, 0.99995, 1.0, 1.0] * 2).reshape(1, 1, 2, 4),
        ),
    ],
)
def test_positional_cos_embeddings(positions, dim, expected_embeddings):
    embeddings = cos_embeddings(pos=positions, dim=dim)
    assert embeddings.shape == expected_embeddings.shape
    np.testing.assert_allclose(np.array(embeddings), expected_embeddings, rtol=1e-4, atol=0)
