import pytest
import haiku as hk
import jax.numpy as jnp

from haiku_geometric.utils import eigv_laplacian
from haiku_geometric.posenc import LaplacianEncoder

@pytest.mark.parametrize('model_type', ['Transformer', 'DeepSet'])
def test_laplacian_encoder(model_type):
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
    )

    def forward_fn(senders, receivers, eigenvalues, eigenvectors, is_training):
        model = LaplacianEncoder(
            8,
            model_type,
            model_dropout=0.1,
            layers=2,
            heads=4,
            post_layers=2,
            norm='batchnorm',
            norm_decay=0.9)
        return model(eigenvalues, eigenvectors, is_training)

    senders = jnp.array(senders)
    receivers = jnp.array(receivers)

    forward = hk.transform_with_state(forward_fn)
    key = hk.PRNGSequence(42)
    params, state = forward.init(next(key), senders, receivers, eigenvalues, eigenvectors, True)
    out, state = forward.apply(params, state, next(key), senders, receivers, eigenvalues, eigenvectors, True)
    assert out.shape == (10, 8)
