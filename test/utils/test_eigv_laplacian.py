import pytest
import jax.numpy as jnp

from haiku_geometric.utils import eigv_laplacian

@pytest.mark.parametrize('eigv_norm', [None, 'L1', 'L2', 'abs-max', 'wavelength', 'wavelength-asin', 'wavelength-soft'])
def test_laplacian(eigv_norm):
    # create synthetic graph
    senders = []
    receivers = []
    weights = []
    nodes = set()

    NODES = 10

    for i in range(NODES - 1):
        senders.append(i)
        receivers.append(i + 1)
        nodes.add(i)
        nodes.add(i + 1)
        weights.append(1)

    eigenvalues, eigenvectors = eigv_laplacian(
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        edge_weight=None,
        normalization=None,
        num_nodes=None,
        k=5,
        eigv_norm=eigv_norm,
    )

    #TODO: compare against expected eigenvalues and eigenvectors