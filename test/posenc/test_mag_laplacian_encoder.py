import pytest
import haiku as hk
import jax.numpy as jnp

from haiku_geometric.utils import eigv_magnetic_laplacian
from haiku_geometric.posenc import MagLaplacianEncoder

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

    eigenvalues, eigenvectors = eigv_magnetic_laplacian(
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        n_node=len(nodes),
        k=5,
        k_excl=0,
        q=0.25,
        q_absolute=False,
        norm_comps_sep=False,
        l2_norm=True,
        sign_rotate=True,
        use_symmetric_norm=True,
    )

    def forward_fn(senders, receivers, eigenvalues, eigenvectors, is_training):
        model = MagLaplacianEncoder(
            d_model_elem=2,
            d_model_aggr=8,
            num_heads=4,
            n_layers=2,
            dropout=0.2,
            use_gnn=False,
            use_signnet=True,
            concatenate_eigenvalues=True,
            use_attention=True,
        )
        return model(senders, receivers, eigenvalues, eigenvectors, is_training)

    senders = jnp.array(senders)
    receivers = jnp.array(receivers)

    forward = hk.transform_with_state(forward_fn)
    key = hk.PRNGSequence(42)
    params, state = forward.init(next(key), senders, receivers, eigenvalues, eigenvectors, True)
    out, state = forward.apply(params, state, next(key), senders, receivers, eigenvalues, eigenvectors, True)
    assert out.shape == (10, 8)
