import jax
import jax.numpy as jnp
from typing import Optional
from haiku_geometric.utils import num_nodes as _num_nodes
from haiku_geometric.transforms import remove_self_loops



def get_laplacian_matrix(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    edge_weight: jnp.ndarray = None,
    normalization: Optional[str] = None,
    num_nodes: Optional[int] = None,
):
    """Returns the Laplacian of a graph.

    Args:
        senders: The senders of the edges.
        receivers: The receivers of the edges.
        edge_weight: The weight of each edge.
        normalization: The normalization to apply to the Laplacian.
        num_nodes: The number of nodes in the graph.

    Returns:
        The Laplacian of the graph.
    """
    if normalization is not None:
        assert normalization in ["sym", "rw"]
    senders, receivers, edge_weight = remove_self_loops(senders, receivers, edge_weight)

    if edge_weight is None:
        edge_weight = jnp.ones(senders.shape[0])

    N = _num_nodes(senders, receivers, num_nodes)

    adj = jnp.zeros((N, N), dtype=jnp.float32)
    adj = adj.at[(senders, receivers)].set(edge_weight)

    if normalization == "sym":
        # A_norm = -D^{-1/2} A D^{-1/2}.
        D = jnp.diag(jnp.sum(adj, axis=1))
        D_inv_sqrt = jnp.linalg.inv(jnp.sqrt(D))
        L = jnp.eye(N) - jnp.matmul(D_inv_sqrt, jnp.matmul(adj, D_inv_sqrt))
    elif normalization == "rw":
        # A_norm = -D^{-1} A.
        D_inv = jnp.diag(jnp.reciprocal(jnp.sum(adj, axis=1)))
        L = jnp.eye(N) - jnp.matmul(D_inv, adj)
    else:
        # L = I - A_norm.
        L = jnp.diag(jnp.sum(adj, axis=1)) - adj
    return L

