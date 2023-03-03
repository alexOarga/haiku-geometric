import jax.numpy as jnp


def remove_self_loops(senders: jnp.ndarray, receivers: jnp.ndarray, edge_weight: jnp.ndarray):
    """
    Removes self loops from a graph.
    """
    mask = senders != receivers
    senders = senders[mask]
    receivers = receivers[mask]
    if edge_weight is not None:
        edge_weight = edge_weight[mask]
    return senders, receivers, edge_weight