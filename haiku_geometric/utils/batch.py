# This was directly adapted from https://github.com/deepmind/jraph/blob/master/jraph/_src/utils.py#L424#L477
# TODO: Future releases might drop support for DataGraphTuple and use directly jraph.
from haiku_geometric.datasets.base import DataGraphTuple
from typing import Sequence, Tuple
import jax.numpy as jnp
import jax.tree_util as tree


def batch(graphs: Sequence[DataGraphTuple]) -> Tuple[DataGraphTuple, jnp.ndarray]:
    r""" Batch a list of graphs into a single graph.

    Args:
        graphs: List of :class:`haiku_geometric.datasets.base.DataGraphTuple`.

    Returns:
        - A single :class:`haiku_geometric.datasets.base.DataGraphTuple` containing the batched graphs.
        - A `jax.numpy.ndarray` with indices indicating to which graph each node belongs to.
    """
    offsets = jnp.cumsum(
      jnp.array([0] + [jnp.sum(g.n_node) for g in graphs[:-1]]))

    def _map_concat(nests):
        concat = lambda *args: jnp.concatenate(args)
        return tree.tree_map(concat, *nests)

    data = DataGraphTuple(
        nodes=_map_concat([g.nodes for g in graphs]),
        edges=_map_concat([g.edges for g in graphs]),
        receivers=jnp.concatenate([g.receivers + offset for g, offset in zip(graphs, offsets)]),
        senders=jnp.concatenate([g.senders + offset for g, offset in zip(graphs, offsets)]),
        n_node=jnp.concatenate([g.n_node for g in graphs]),
        n_edge=jnp.concatenate([g.n_edge for g in graphs]),
        globals=_map_concat([g.globals for g in graphs]),
        position=_map_concat([g.position for g in graphs]),
        y=_map_concat([g.y for g in graphs]),
        train_mask=_map_concat([g.train_mask for g in graphs]))
    batch_index = jnp.repeat(jnp.arange(len(graphs)), repeats=data.n_node)

    return data, batch_index


def unbatch(graph: DataGraphTuple) -> Sequence[DataGraphTuple]:
    r""" Unbatch a graph into a list of graphs.

    Args:
        graph: A graph :class:`haiku_geometric.datasets.base.DataGraphTuple` to unbatch.

    Returns:
        A list of :class:`haiku_geometric.datasets.base.DataGraphTuple` containing the unbatched graphs.
    """
    def _map_split(nests, indices):
        if isinstance(indices, int):
            n_lists = indices
        else:
            n_lists = len(indices) + 1
        concat = lambda field: jnp.split(field, indices)
        nest_of_lists = tree.tree_map(concat, nests)
        list_of_nests = [
            tree.tree_map(lambda _, x: x[i], nests, nest_of_lists)
            for i in range(n_lists)
        ]
        return list_of_nests

    all_n_node = graph.n_node[:, None]
    all_n_edge = graph.n_edge[:, None]
    node_offsets = jnp.cumsum(graph.n_node[:-1])
    all_nodes = _map_split(graph.nodes, node_offsets)
    edge_offsets = jnp.cumsum(graph.n_edge[:-1])
    all_edges = _map_split(graph.edges, edge_offsets)
    all_globals = _map_split(graph.globals, len(graph.n_node))
    all_position = _map_split(graph.position, node_offsets)
    all_y = _map_split(graph.y, node_offsets)
    all_train_mask = _map_split(graph.train_mask, len(graph.n_node))
    all_senders = jnp.split(graph.senders, edge_offsets)
    all_receivers = jnp.split(graph.receivers, edge_offsets)

    # correct offset on senders and receivers
    n_graphs = graph.n_node.shape[0]
    for graph_index in jnp.arange(n_graphs)[1:]:
        all_senders[graph_index] -= node_offsets[graph_index - 1]
        all_receivers[graph_index] -= node_offsets[graph_index - 1]

    return [
        DataGraphTuple._make(elements)
        for elements in zip(all_nodes, all_edges, all_receivers, all_senders,
                            all_n_node, all_n_edge, all_globals, all_position,
                            all_y, all_train_mask)
    ]
