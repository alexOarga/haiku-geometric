import jax.numpy as jnp

from haiku_geometric.utils import get_laplacian, get_laplacian_matrix
from haiku_geometric.datasets.base import DataGraphTuple

def test_laplacian():
    # create synthetic graph
    data = DataGraphTuple(
        nodes=jnp.array([0, 0, 0]),
        senders=jnp.array([0, 0, 1, 1, 1, 2, 2, 2]),
        receivers=jnp.array([1, 2, 0, 1, 2, 0, 1, 2]),
        edges=jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
        n_node=4,
        n_edge=5,
        globals=jnp.array([0.0, 0.0, 0.0]),
        position=None,
        y=jnp.array([0.0, 0.0, 0.0, 0, 0]),
        train_mask=jnp.array([True, True, True, False]),
    )

    L = get_laplacian_matrix(data.senders, data.receivers, data.edges)
    # Note the expected values are the out-degree
    expected_L = jnp.array([[ 3., -1., -2.], [-3.,  8., -5.], [-6., -7., 13.]])
    assert jnp.allclose(L, expected_L)

    L = get_laplacian_matrix(data.senders, data.receivers, data.edges, normalization = 'sym')
    expected_L = jnp.array([[1., -0.20412414, -0.3202563 ], [-0.6123724, 1., -0.4902903 ], [-0.9607689, -0.6864065, 1.]])
    assert jnp.allclose(L, expected_L)

    L = get_laplacian_matrix(data.senders, data.receivers, data.edges, normalization = 'rw')
    expected_L = jnp.array([[1., -0.33333334, -0.6666667], [-0.375, 1., -0.625], [-0.4615385, -0.53846157, 1.]])
    assert jnp.allclose(L, expected_L)

    senders, receivers, edge_weight = get_laplacian(data.senders, data.receivers, data.edges)
    expected_senders = jnp.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    expected_receivers = jnp.array([1, 2, 0, 2, 0, 1, 0, 1, 2])
    expected_weight = jnp.array([-1., -2., -3., -5., -6., -7., 3., 8., 13.])
    assert jnp.allclose(senders, expected_senders)
    assert jnp.allclose(receivers, expected_receivers)
    assert jnp.allclose(edge_weight, expected_weight)

    senders, receivers, edge_weight = get_laplacian(data.senders, data.receivers, data.edges, normalization='sym')
    expected_senders = jnp.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    expected_receivers = jnp.array([1, 2, 0, 2, 0, 1, 0, 1, 2])
    expected_weight = jnp.array([-0.2041, -0.3203, -0.6124, -0.4903, -0.9608, -0.6864, 1.0000, 1.0000, 1.0000])
    assert jnp.allclose(senders, expected_senders)
    assert jnp.allclose(receivers, expected_receivers)
    assert jnp.allclose(edge_weight, expected_weight, rtol=1e-04, atol=1e-04)

    senders, receivers, edge_weight = get_laplacian(data.senders, data.receivers, data.edges, normalization='rw')
    expected_senders = jnp.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    expected_receivers = jnp.array([1, 2, 0, 2, 0, 1, 0, 1, 2])
    expected_weight = jnp.array([-0.3333, -0.6667, -0.3750, -0.6250, -0.4615, -0.5385,  1.0000,  1.0000, 1.0000])
    assert jnp.allclose(senders, expected_senders)
    assert jnp.allclose(receivers, expected_receivers)
    assert jnp.allclose(edge_weight, expected_weight, rtol=1e-04, atol=1e-04)

