# This code was adapted from or completely taken from: https://github.com/deepmind/digraph_transformer

import jax.numpy as jnp
import numba
import numpy as np


EPS = 1E-8


# Necessary to work around numbas limitations with specifying axis in norm and
# braodcasting in parallel loops.
@numba.njit('float64[:, :](float64[:, :])', parallel=False)
def _norm_2d_along_first_dim_and_broadcast(array):
    """Equivalent to `linalg.norm(array, axis=0)[None, :] * ones_like(array)`."""
    output = np.zeros(array.shape, dtype=array.dtype)
    for i in numba.prange(array.shape[-1]):
        output[:, i] = np.linalg.norm(array[:, i])
    return output


# Necessary to work around numbas limitations with specifying axis in norm and
# braodcasting in parallel loops.
@numba.njit('float64[:, :](float64[:, :])', parallel=False)
def _max_2d_along_first_dim_and_broadcast(array):
    """Equivalent to `array.max(0)[None, :] * ones_like(array)`."""
    output = np.zeros(array.shape, dtype=array.dtype)
    for i in numba.prange(array.shape[-1]):
        output[:, i] = array[:, i].max()
    return output


def fill_diagonal(a, val):
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    a = a.at[..., i, j].set(val)
    return a


def eigv_magnetic_laplacian_numba(
        senders: jnp.ndarray, receivers: jnp.ndarray, n_node: jnp.ndarray,
        k: int, k_excl: int, q: float, q_absolute: bool,
        norm_comps_sep: bool, l2_norm: bool, sign_rotate: bool,
        use_symmetric_norm: bool):
    r""" k non-ptrivial complex eigenvectors of the smallest k eigenvectors of the magnetic laplacian.
  Args:
    senders: Origin of the edges of shape [m].
    receivers: Target of the edges of shape [m].
    n_node: array shape [2]
    padded_nodes_size: int the number of nodes including padding.
    k: Returns top k eigenvectors.
    k_excl: The top (trivial) eigenvalues / -vectors to exclude.
    q: Factor in magnetic laplacian. Default 0.25.
    q_absolute: If true `q` will be used, otherwise `q / m_imag / 2`.
    norm_comps_sep: If true first imaginary part is separately normalized.
    l2_norm: If true we use l2 normalization and otherwise the abs max value.
    sign_rotate: If true we decide on the sign based on max real values and
      rotate the imaginary part.
    use_symmetric_norm: symmetric (True) or row normalization (False).
  Returns:
    array of shape [<= k] containing the k eigenvalues.
    array of shape [n, <= k] containing the k eigenvectors.
    array of shape [n, n] the laplacian.
  """
    # Handle -1 padding

    adj = jnp.zeros(int(n_node * n_node), dtype=jnp.float64)
    linear_index = receivers + (senders * n_node).astype(senders.dtype)
    adj = adj.at[linear_index].set(1)
    adj = adj.reshape(n_node, n_node)
    adj = jnp.where(adj > 1, 1, adj)

    symmetric_adj = adj + adj.T
    symmetric_adj = jnp.where((adj != 0) & (adj.T != 0), symmetric_adj / 2,
                              symmetric_adj)

    symmetric_deg = symmetric_adj.sum(-2)

    if not q_absolute:
        m_imag = (adj != adj.T).sum() / 2
        m_imag = min(m_imag, n_node)
        q = q / (m_imag if m_imag > 0 else 1)

    theta = 1j * 2 * jnp.pi * q * (adj - adj.T)

    if use_symmetric_norm:
        inv_deg = jnp.zeros((n_node, n_node), dtype=jnp.float64)
        inv_deg = fill_diagonal(
            inv_deg, 1. / jnp.sqrt(jnp.where(symmetric_deg < 1, 1, symmetric_deg)))
        eye = jnp.eye(n_node)
        inv_deg = inv_deg.astype(adj.dtype)
        deg = inv_deg @ symmetric_adj.astype(adj.dtype) @ inv_deg
        laplacian = eye - deg * jnp.exp(theta)

        # mask = jnp.arange(padded_nodes_size) < n_node[:1]
        # mask = jnp.expand_dims(mask, -1) & jnp.expand_dims(mask, 0)
        # laplacian = mask.astype(adj.dtype) * laplacian
    else:
        deg = jnp.zeros((n_node, n_node), dtype=jnp.float64)
        deg = fill_diagonal(deg, symmetric_deg)
        laplacian = deg - symmetric_adj * jnp.exp(theta)

    eigenvalues, eigenvectors = jnp.linalg.eigh(laplacian)

    eigenvalues = eigenvalues[..., k_excl:k_excl + k]
    eigenvectors = eigenvectors[..., k_excl:k_excl + k]

    if sign_rotate:
        sign = jnp.zeros((eigenvectors.shape[1],), dtype=eigenvectors.dtype)
        for i in range(eigenvectors.shape[1]):
            argmax_i = jnp.abs(eigenvectors[:, i].real).argmax()
            sign = sign.at[i].set(jnp.sign(eigenvectors[argmax_i, i].real))
        eigenvectors = jnp.expand_dims(sign, 0) * eigenvectors

        argmax_imag_0 = eigenvectors[:, 0].imag.argmax()
        rotation = jnp.angle(eigenvectors[argmax_imag_0:argmax_imag_0 + 1])
        eigenvectors = eigenvectors * jnp.exp(-1j * rotation)

    if norm_comps_sep:
        # Only scale eigenvectors that seems to be more than numerical errors
        eps = EPS / jnp.sqrt(eigenvectors.shape[0])
        if l2_norm:
            scale_real = _norm_2d_along_first_dim_and_broadcast(jnp.real(eigenvectors))
            real = jnp.real(eigenvectors) / scale_real
        else:
            scale_real = _max_2d_along_first_dim_and_broadcast(
                jnp.abs(jnp.real(eigenvectors)))
            real = jnp.real(eigenvectors) / scale_real
        scale_mask = jnp.abs(
            jnp.real(eigenvectors)).sum(0) / eigenvectors.shape[0] > eps
        eigenvectors = eigenvectors.at[:, scale_mask].set(
            real[:, scale_mask] + 1j * jnp.imag(eigenvectors)[:, scale_mask])

        if l2_norm:
            scale_imag = _norm_2d_along_first_dim_and_broadcast(jnp.imag(eigenvectors))
            imag = jnp.imag(eigenvectors) / scale_imag
        else:
            scale_imag = _max_2d_along_first_dim_and_broadcast(
                jnp.abs(jnp.imag(eigenvectors)))
            imag = jnp.imag(eigenvectors) / scale_imag
        scale_mask = jnp.abs(
            jnp.imag(eigenvectors)).sum(0) / eigenvectors.shape[0] > eps
        eigenvectors = eigenvectors.at[:, scale_mask].set(
            jnp.real(eigenvectors)[:, scale_mask] + 1j * imag[:, scale_mask])
    elif not l2_norm:
        scale = _max_2d_along_first_dim_and_broadcast(
            np.array(jnp.absolute(eigenvectors), dtype=np.float64))
        eigenvectors = eigenvectors / scale

    return eigenvalues.real, eigenvectors, laplacian


def eigv_magnetic_laplacian(
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        n_node: int,
        k: int,
        k_excl: int,
        q: float = 0.25,
        q_absolute: bool = False,
        norm_comps_sep: bool = False,
        l2_norm: bool = True,
        sign_rotate: bool = True,
        use_symmetric_norm: bool = False,
        # ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
):
    """ k non-ptrivial *complex* eigenvectors of the smallest k eigenvectors of the magnetic laplacian.
    This implementation is from the paper `"Transformers Meet Directed Graphs" <https://arxiv.org/abs/2302.00049>`_ paper

  Args:
    senders (jnp.ndarray): Origin of the edges of shape [m].
    receivers (jnp.ndarray): Target of the edges of shape [m].
    n_node (int): Number of nodes in the graph.
    k (int): Returns top k eigenvectors.
    k_excl (int): The top (trivial) eigenvalues / -vectors to exclude.
    q (float, optional): Factor in magnetic laplacian.
        (default: :obj:`0.25`)
    q_absolute (bool, optional): If true `q` will be used, otherwise `q / m_imag / 2`.
        (default: :obj:`False`)
    norm_comps_sep (bool, optional): If true first imaginary part is separately normalized.
        (default: :obj:`False`)
    l2_norm (bool, optional): If true we use l2 normalization and otherwise the abs max value.
      Will be treated as false if `norm_comps_sep` is true.
      (default: :obj:`True`)
    sign_rotate (bool, optional): If true we decide on the sign based on max real values and
      rotate the imaginary part.
      (default: :obj:`True`)
    use_symmetric_norm (bool, optional): symmetric (True) or row normalization (False).
        (default: :obj:`False`)
  Returns:
    - :obj:`(jnp.float64)` list with arrays of shape [<= k] containing the k eigenvalues.
    - :obj:`(jnp.complex128)` list with arrays of shape [n_node, <= k] containing the k eigenvectors.
  """
    eigenvalues = jnp.zeros((k), dtype=jnp.float64)
    eigenvectors = jnp.zeros((n_node, k), dtype=jnp.complex128)

    eigenvalue, eigenvector, _ = eigv_magnetic_laplacian_numba(
        senders,
        receivers,
        n_node,
        k=k,
        k_excl=k_excl,
        q=q,
        q_absolute=q_absolute,
        norm_comps_sep=norm_comps_sep,
        l2_norm=l2_norm,
        sign_rotate=sign_rotate,
        use_symmetric_norm=use_symmetric_norm)

    eigenvalues = eigenvalues.at[:eigenvalue.shape[0]].set(eigenvalue)
    eigenvectors = eigenvectors.at[:eigenvector.shape[0], :eigenvector.shape[1]].set(eigenvector)
    return eigenvalues, eigenvectors