import jax.numpy as jnp
import jraph
from typing import Optional


def validate_input(
        nodes: jnp.ndarray = None,
        senders: jnp.ndarray = None,
        receivers: jnp.ndarray = None,
        edges: Optional[jnp.ndarray] = None,
        graph: Optional[jraph.GraphsTuple] = None):
    if graph is not None:
        nodes, edges, receivers, senders, _, _, _ = graph
    elif nodes is None or senders is None or receivers is None:
        raise ValueError("Input parameters 'nodes', 'senders' and 'receivers' are required. " +
                         "You can also provide this parameters through the 'graph' parameter.")
    return nodes, edges, receivers, senders
