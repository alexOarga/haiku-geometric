import jax
import jax.numpy as jnp
from jax.ops import segment_sum
from typing import Optional
from haiku_geometric.utils import num_nodes as _num_nodes
from haiku_geometric.transforms import add_self_loops, remove_self_loops



def get_laplacian_matrix(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    edge_weight: jnp.ndarray = None,
    normalization: Optional[str] = None,
    num_nodes: Optional[int] = None,
):
    r"""Returns the Laplacian of a graph.

    Args:
        senders (jnp.ndarray): The senders of the edges.
        receivers (jnp.ndarray): The receivers of the edges.
            (default: :obj:`None`)
        edge_weight (jnp.ndarray): The weight of each edge.
            (default: :obj:`None`)
        normalization (str, optional) : The normalization to apply to the Laplacian.
            (default: :obj:`None`). Available options are:

            1. :obj:`None` : No normalization.
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"` : Symmetric normalization.
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}`

            3. :obj:`"rw"` : Random-walk normalization.
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

        num_nodes (int, optional): The number of nodes in the graph.
            (default: :obj:`None`)

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
        # L = D - A
        L = jnp.diag(jnp.sum(adj, axis=1)) - adj
    return L


# Adapted from Pytorch Geoemtric: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/get_laplacian.html
def get_laplacian(
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    edge_weight: jnp.ndarray = None,
    normalization: Optional[str] = None,
    num_nodes: Optional[int] = None,
):
    r"""Returns the Laplacian of a graph. Unlike :func:`get_laplacian_matrix`, this
    function performs the operations over the indices of the graph, rather than the
    adjacency matrix. Consequently, this function returns the indices and weights
    of the Laplacian.

    Args:
        senders (jnp.ndarray): The senders of the edges.
        receivers (jnp.ndarray): The receivers of the edges.
            (default: :obj:`None`)
        edge_weight (jnp.ndarray): The weight of each edge.
            (default: :obj:`None`)
        normalization (str, optional) : The normalization to apply to the Laplacian.
            (default: :obj:`None`). Available options are:

            1. :obj:`None` : No normalization.
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"` : Symmetric normalization.
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}`

            3. :obj:`"rw"` : Random-walk normalization.
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

        num_nodes (int, optional): The number of nodes in the graph.
            (default: :obj:`None`)

    Returns:
        Tuple(jnp.ndarray, jnp.ndarray, jnp.ndarray): Senders, receivers and weights of the Laplacian.
    """
    if normalization is not None:
        assert normalization in ["sym", "rw"]
    senders, receivers, edge_weight = remove_self_loops(senders, receivers, edge_weight)

    if edge_weight is None:
        edge_weight = jnp.ones(senders.shape[0])

    N = _num_nodes(senders, receivers, num_nodes)
    deg = segment_sum(edge_weight, senders, num_segments=N)

    if normalization is None:
        # L = D - A
        senders, receivers, _ = add_self_loops(senders, receivers, num_nodes=N)
        edge_weight = jnp.concatenate([-edge_weight, deg], axis=0)
    elif normalization == "sym":
        # A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = jnp.power(deg, -0.5)
        deg_inv_sqrt = jnp.where(jnp.isinf(deg_inv_sqrt), 0, deg_inv_sqrt)
        edge_weight = deg_inv_sqrt[senders] * edge_weight * deg_inv_sqrt[receivers]
        senders, receivers, edge_weight = add_self_loops(senders, receivers, -edge_weight,
                                                         fill_value=1.0, num_nodes=N)
    elif normalization == "rw":
        deg_inv = 1.0 / deg
        deg_inv = jnp.where(jnp.isinf(deg_inv), 0, deg_inv)
        edge_weight = deg_inv[senders] * edge_weight

        senders, receivers, edge_weight = add_self_loops(senders, receivers, -edge_weight,
                                                            fill_value=1.0, num_nodes=N)

    return senders, receivers, edge_weight