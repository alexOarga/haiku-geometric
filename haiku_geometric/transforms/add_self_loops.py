import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
from typing import Tuple


def add_self_loops(
        receivers: jnp.ndarray, senders: jnp.ndarray,
        total_num_nodes: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Adds self edges. Assumes self edges are not in the graph yet.

    Args:
        receivers (jnp.ndarray): Array of receiver node indices.
        senders (jnp.ndarray): Array of sender node indices.
        total_num_nodes (int): Total number of nodes in the graph.
    """
    receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)), axis=0)
    senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)), axis=0)
    return receivers, senders
