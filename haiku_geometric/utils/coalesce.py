import jax
import jax.numpy as jnp
from jax.ops import segment_sum
from typing import Optional, Tuple, Union
from haiku_geometric.utils import num_nodes as _num_nodes


# Adapted from Pytorch Geometric
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/coalesce.html#coalesce
def coalesce(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    edge_attr: Optional[jnp.ndarray] = None,
    num_nodes: Optional[int] = None,
    #reduce: str = "add", # TODO: add 'scatter' method and include this parameter
    is_sorted: bool = False,
    sort_by_row: bool = True,
): #-> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Returns the unique edges in a graph.

    Args:
        senders (jnp.ndarray): The senders of the edges.
        receivers (jnp.ndarray): The receivers of the edges.
        edge_attr (jnp.ndarray, optional): The edge attributes.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes in the graph.
            (default: :obj:`None`)
        is_sorted (bool, optional): Whether senders and receiver are sorted row-wise.
            (default: :obj:`False`)
        sort_by_row (bool, optional): Whether to sort the edges by row. If :obj:`False`,
            the edges will be sorted by column.
            (default: :obj:`True`)

    Returns:
        :obj:`Tuple(jnp.ndarray, jnp.ndarray)` with senders, receivers if :obj:`edge_attr` is :obj:`None`.
        :obj:`Tuple(jnp.ndarray, jnp.ndarray, jnp.ndarray)` with senders, receivers, edge_attr otherwise.
    """

    # Create a unique edge index to reproduce the PyG implementation
    edge_index = jnp.stack([senders, receivers], axis=0)
    num_edges = senders.shape[0]
    num_nodes = _num_nodes(senders, receivers, num_nodes)

    idx = jnp.empty(num_edges + 1)
    idx = idx.at[0].set(-1)
    idx = idx.at[1:].set(edge_index[1 - int(sort_by_row)])
    idx = idx.at[1:].set(idx[1:] * num_nodes + edge_index[int(sort_by_row)])
    if not is_sorted:
        #idx = jnp.sort(idx)
        perm = jnp.argsort(idx[1:])
        aux = idx[1:][perm]
        idx = idx.at[1:].set(aux)
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm] if edge_attr is not None else None

    mask = idx[1:] > idx[:-1]
    if jnp.all(mask):
        if edge_attr is not None:
            return edge_index[0], edge_index[1], edge_attr
        else:
            return edge_index[0], edge_index[1]

    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index[0], edge_index[1]

    dim_size = edge_index.shape[1]
    idx = jnp.arange(num_edges)
    idx -= jnp.cumsum(jnp.logical_not(mask), axis=0)
    edge_attr = segment_sum(edge_attr, idx, num_segments=dim_size) # Equivalent to scatter add
    return edge_index[0], edge_index[1], edge_attr