import jax
import jax.numpy as jnp
from jax.ops import segment_sum
from typing import Optional, Tuple, Union
from haiku_geometric.utils import coalesce


def to_undirected(
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        edge_attr: Optional[jnp.ndarray] = None,
        num_nodes: Optional[int] = None,
): #-> Union[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""
    Returns the undirected version of a graph.

    Args:
        senders (jnp.ndarray): The senders of the edges.
        receivers (jnp.ndarray): The receivers of the edges.
        edge_attr (jnp.ndarray, optional): The edge attributes.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes in the graph.
            (default: :obj:`None`)

    Returns:
        :obj:`Tuple(jnp.ndarray, jnp.ndarray)` with senders, receivers if :obj:`edge_attr` is :obj:`None`.
        :obj:`Tuple(jnp.ndarray, jnp.ndarray, jnp.ndarray)` with senders, receivers, edge_attr otherwise.
    """

    senders, receivers = \
        jnp.concatenate([senders, receivers], axis=-1), jnp.concatenate([receivers, senders], axis=-1)
    edge_attr = jnp.concatenate([edge_attr, edge_attr], axis=0) if edge_attr is not None else None
    return coalesce(senders, receivers, edge_attr, num_nodes)