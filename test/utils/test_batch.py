import jax.numpy as jnp
from haiku_geometric.utils import batch, unbatch
from haiku_geometric.datasets.base import DataGraphTuple

def test_batch():
    data1 = DataGraphTuple(
        nodes=jnp.array([[1, 2], [2, 3], [3, 4], [4, 5]]),
        senders=jnp.array([0, 0, 1, 1, 1, 2, 2, 2, 2]),
        receivers=jnp.array([1, 2, 0, 1, 2, 0, 1, 2, 3]),
        edges=jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        n_node=jnp.array([4]),
        n_edge=jnp.array([9]),
        globals=jnp.array([1.0, 2.0, 3.0]),
        position=None,
        y=jnp.array([0.0, 1.0, 0.0, 1.0]),
        train_mask=jnp.array([True, True, True, False]),
    )

    data2 = DataGraphTuple(
        nodes=jnp.array([[0, 0], [0, 0], [1, 1]]),
        senders=jnp.array([0, 0, 1, 1, 1, 2, 2, 2]),
        receivers=jnp.array([1, 2, 0, 1, 2, 0, 1, 2]),
        edges=jnp.array([1, 2, 3, 4, 5, 6, 7, 8]),
        n_node=jnp.array([3]),
        n_edge=jnp.array([8]),
        globals=jnp.array([0.0, 0.0, 0.0]),
        position=None,
        y=jnp.array([0.0, 0.0, 0.0, 0, 0]),
        train_mask=jnp.array([True, True, True, False]),
    )

    new_data = batch([data1, data2])

    assert jnp.all(new_data.nodes == jnp.array([[1, 2], [2, 3], [3, 4], [4, 5], [0, 0], [0, 0], [1, 1]]))
    assert jnp.all(new_data.edges == jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8]))
    assert jnp.all(new_data.receivers == jnp.array([1, 2, 0, 1, 2, 0, 1, 2, 3, 5, 6, 4, 5, 6, 4, 5, 6]))
    assert jnp.all(new_data.senders == jnp.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 4, 4, 5, 5, 5, 6, 6, 6]))
    assert jnp.all(new_data.n_node == jnp.array([4, 3]))
    assert jnp.all(new_data.n_edge == jnp.array([9, 8]))
    assert jnp.all(new_data.globals == jnp.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]]))
    assert jnp.all(new_data.position == None)
    assert jnp.all(new_data.y == jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0, 0]))
    assert jnp.all(new_data.train_mask == jnp.array([True, True, True, False, True, True, True, False]))

    unbatch_graphs_list = unbatch(new_data)
    assert len(unbatch_graphs_list) == 2

    graph1 = unbatch_graphs_list[0]
    graph2 = unbatch_graphs_list[1]
    assert jnp.all(graph1.nodes == data1.nodes)
    assert jnp.all(graph1.edges == data1.edges)
    assert jnp.all(graph1.receivers == data1.receivers)
    assert jnp.all(graph1.senders == data1.senders)
    assert jnp.all(graph1.n_node == data1.n_node)
    assert jnp.all(graph1.n_edge == data1.n_edge)
    assert jnp.all(graph1.globals == data1.globals)
    assert jnp.all(graph1.position == data1.position)
    assert jnp.all(graph1.y == data1.y)
    assert jnp.all(graph1.train_mask == data1.train_mask)

    assert jnp.all(graph2.nodes == data2.nodes)
    assert jnp.all(graph2.edges == data2.edges)
    assert jnp.all(graph2.receivers == data2.receivers)
    assert jnp.all(graph2.senders == data2.senders)
    assert jnp.all(graph2.n_node == data2.n_node)
    assert jnp.all(graph2.n_edge == data2.n_edge)
    assert jnp.all(graph2.globals == data2.globals)
    assert jnp.all(graph2.position == data2.position)
    assert jnp.all(graph2.y == data2.y)
    assert jnp.all(graph2.train_mask == data2.train_mask)