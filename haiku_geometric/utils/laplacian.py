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
        D = jnp.diag(jnp.sum(adj, axis=1))
        L = D - adj
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
        :obj:`Tuple(jnp.ndarray, jnp.ndarray, jnp.ndarray)`: Senders, receivers and weights of the Laplacian.
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

# This function was directly adapted from GraphGPS repository:
# https://github.com/rampasek/GraphGPS
def eigv_normalizer(eigenvectors, eigenvalues, normalization="L2", eps=1e-12):
    r"""
    Normalizes eigenvectors and eigenvalues.

    Args:
        eigenvectors (jnp.ndarray): Input eigenvectors.
        eigenvalues (jnp.ndarray): Input eigenvalues.
        normalization (str, optional): Normalization to apply to the eigenvectors. Available options are: :obj:`None`, :obj:`"L1"`,  :obj:`"L2"`, :obj:`"abs-max"`,  :obj:`"wavelength"`,  :obj: `"wavelength-asin"` and :obj:`"wavelength-soft"`.
            (default: :obj:`"L2"`)
        eps (float, optional): Small value to avoid division by zero.
            (default: :obj:`1e-12`)
    Returns:
        :obj:`jnp.ndarray`: Normalized eigenvectors.
    """
    #eigenvalues = jnp.expand_dims(eigenvalues, axis=0)

    if normalization is None:
        return eigenvectors
    if normalization == "L1":
        denom = jnp.linalg.norm(eigenvectors, ord=1, axis=0)
    elif normalization == "L2":
        denom = jnp.linalg.norm(eigenvectors, ord=2, axis=0)
    elif normalization == "abs-max":
        denom = jnp.max(jnp.abs(eigenvectors), axis=0)
    elif normalization == "wavelength":
        denom = jnp.max(jnp.abs(eigenvectors), axis=0)
        eigval_denom = jnp.sqrt(eigenvalues)
        eigval_denom = eigval_denom.at[eigval_denom < eps].set(1)
        denom = denom * eigval_denom * 2 / jnp.pi
    elif normalization == "wavelength-asin":
        denom_temp = jnp.max(jnp.abs(eigenvectors), axis=0).clip(min=eps)
        eigenvectors = jnp.arcsin(eigenvectors / denom_temp)
        eigenval_denom = jnp.sqrt(eigenvalues)
        eigenval_denom = eigenval_denom.at[eigenval_denom < eps].set(1)
        denom = eigenval_denom
    elif normalization == "wavelength-soft":
        denom = jax.nn.softmax(jnp.abs(eigenvectors), axis=0)* jnp.abs(eigenvectors).sum(axis=0, keepdims=True)
        eigval_denom = jnp.sqrt(eigenvalues)
        eigval_denom= eigval_denom.at[eigval_denom < eps].set(1)
        denom = denom * eigval_denom
    else:
        raise ValueError("Unknown normalization: {}".format(normalization))

    denom = denom.clip(min=eps)
    eigenvectors = eigenvectors / denom
    return eigenvectors

def eigv_laplacian(
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        edge_weight: jnp.ndarray = None,
        normalization: Optional[str] = None,
        num_nodes: Optional[int] = None,
        k: int = 5,
        eigv_norm: Optional[str] = 'L2',
):
    r"""Returns the top-k eigenvectors of the Laplacian of a graph.

    Args:
        senders (jnp.ndarray): The senders of the edges.
        receivers (jnp.ndarray): The receivers of the edges.
        edge_weight (jnp.ndarray): The weight of each edge.
            (default: :obj:`None`)
        normalization (str, optional) : The normalization to apply to the Laplacian.
            (default: :obj:`None`). Available options are:
            :obj:`None`, :obj:`"sym"` , :obj:`"rw"`.
        num_nodes (int, optional): The number of nodes in the graph.
            (default: :obj:`None`)
        k (int, optional): The number of eigenvectors to return.
            (default: :obj:`5`)
        eigv_norm (str, optional): The normalization to apply to the eigenvectors.
            (default: :obj:`"L2"`). Available options are:
            1. :obj:`None` : No normalization.
            2. :obj:`"L2"` : Normalize the eigenvectors to have unit L2 norm.

    Returns:
        :obj:`(jnp.ndarray)`:  (k,) eigenvalues.
        :obj:`(jnp.ndarray)`:  (num_nodes, k) of eigenvector values per node.
    """

    L = get_laplacian_matrix(
        senders, receivers, edge_weight, normalization, num_nodes)
    evals, evects = jnp.linalg.eigh(L)

    N = len(evals)
    idx = (jnp.argsort(evals))[:k]
    evals, evects = evals[idx], jnp.real(evects[:, idx])
    evals = jnp.clip(jnp.real(evals), a_min=0)
    evects = evects.astype(jnp.float32)
    evects = eigv_normalizer(evects, evals, normalization=eigv_norm)

    if N < k:
        # pad on last dimension with nan
        npad = [(0, 0)] * evects.ndim
        npad[-1] = (0, k - N)
        evects = jnp.pad(evects, npad, 'constant', constant_values=jnp.nan)
        npad = [(0, 0)] * evals.ndim
        npad[-1] = (0, k - N)
        evals = jnp.pad(evals, npad, 'constant', constant_values=jnp.nan)

    return evals, evects