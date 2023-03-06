import jax.numpy as jnp

from haiku_geometric.utils import coalesce
from haiku_geometric.datasets.base import DataGraphTuple

def test_coalesce():

    senders = jnp.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    receivers = jnp.array([1, 2, 0, 1, 2, 0, 1, 2, 0])
    weights = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    out_senders, out_receivers, out_weights = coalesce(senders, receivers, weights)
    expected_senders = jnp.array([0, 0, 1, 1, 1, 2, 2, 2])
    expected_receivers = jnp.array([1, 2, 0, 1, 2, 0, 1, 2])
    expected_weights = jnp.array([1., 2., 3., 4., 5., 15., 7., 8.])
    assert jnp.array_equal(out_senders, expected_senders)
    assert jnp.array_equal(out_receivers, expected_receivers)
    assert jnp.array_equal(out_weights, expected_weights)
