import pytest
import haiku as hk
import jax.numpy as jnp
import jax
import optax

from haiku_geometric.datasets import KarateClub
from haiku_geometric.models import Node2Vec


@pytest.mark.parametrize('p, q', [
    (1.0, 1.0),  # uniform sampling
    (0.5, 0.4),  # biased sampling
])
def test_node2vec(p, q):

    dataset = KarateClub()
    data = dataset.data[0]
    EMB_DIM = 128

    def forward():
        model = Node2Vec(data.senders, data.receivers, embedding_dim=EMB_DIM, walk_length=20,
                    context_size=10, walks_per_node=10,
                    num_negative_samples=1, p=p, q=q)
        return model()

    model = hk.transform(forward)
    model = hk.without_apply_rng(model)
    rng = jax.random.PRNGKey(42)
    num_nodes = jnp.max(jnp.maximum(data.senders, data.receivers)) + 1
    params = model.init(rng)
    
    opt_init, opt_update = optax.adam(learning_rate=0.1)
    opt_state = opt_init(params)

    #@jax.jit
    def loss_fn(params):
        embedding, loss = model.apply(params)
        assert embedding.shape == (data.n_node, EMB_DIM)
        return loss

    #@jax.jit
    def update(params, opt_state):
        g = jax.grad(loss_fn)(params)
        updates, opt_state = opt_update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state

    # one iteration only
    for step in range(1):
        num_nodes = jnp.max(jnp.maximum(data.senders, data.receivers)) + 1
        params, opt_state = update(params, opt_state)
