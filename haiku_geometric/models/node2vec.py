import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import haiku as hk
from typing import Optional, Tuple
from haiku_geometric.utils import random_walk


class Node2Vec:
    r"""
    """
    def __init__(
        self,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        num_nodes: Optional[int] = None,
        rng = jax.random.PRNGKey(42),
    ):

        N = self._num_nodes(senders, receivers, num_nodes)
        self.num_nodes = N
        self.senders = senders # TODO: recover from BCOO
        self.receivers = receivers
        self.adj = BCOO((
            jnp.ones(senders.shape[0]),
            jnp.stack([senders, receivers], axis=0)),
            shape=(N, N))
        self.EPS = 1e-15
        assert walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.rng = rng


    def _num_nodes(self, senders: jnp.ndarray, receivers: jnp.ndarray, num_nodes: Optional[int] = None):
        if num_nodes is None:
            return jnp.max(jnp.concatenate([senders, receivers])) + 1
        else:
            return num_nodes

    def pos_sample(self, batch: jnp.ndarray):
        """Returns positive samples."""
        #batch = jnp.repeat(batch, self.walks_per_node) # TODO: is batch needed?
        rw = random_walk(self.senders, self.receivers, self.walk_length, self.p, self.q)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return jnp.concatenate(walks, axis=0)

    def neg_sample(self, batch: jnp.ndarray):
        """Returns negative samples."""
        batch = jnp.repeat(batch, self.walks_per_node * self.num_negative_samples)
        rw = jax.random.randint(self.rng, (batch.size, self.walk_length), 0, self.num_nodes)
        rw = jnp.hstack([jnp.reshape(batch, (-1, 1)), rw])
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return jnp.concatenate(walks, axis=0)

    def sample(self, batch: jnp.ndarray):
        pos = self.pos_sample(batch)
        neg = self.neg_sample(batch)
        return pos, neg



class Node2VecTrainer(hk.Module):
    def __init__(self, model: Node2Vec):
        super().__init__()
        self.model = model

        self.embedding_dim = model.embedding_dim
        self.num_nodes = model.num_nodes
        self.EPS = model.EPS

        w_init = hk.initializers.TruncatedNormal()
        self.embedding = hk.get_parameter("embedding", shape=[self.num_nodes, self.embedding_dim], init=w_init)
        self.rng = model.rng

    def __call__(self, pos_rw: jnp.ndarray, neg_rw: jnp.ndarray):
        """ This is a loss function. """
        start, rest = pos_rw[:, 0].astype(int), pos_rw[:, 1:].astype(int)

        jax.debug.print("s: {start}", start=start.shape)
        jax.debug.print("e: {start}", start=self.embedding.shape)
        h_start = self.embedding[start].reshape(pos_rw.shape[0], 1, self.embedding_dim)
        h_rest = self.embedding[rest.reshape(-1)].reshape(pos_rw.shape[0], -1, self.embedding_dim)

        out = (h_start * h_rest).sum(axis=-1).reshape(-1)
        pos_loss = -jnp.log(jax.nn.sigmoid(out) + self.EPS).mean()

        start, rest = neg_rw[:, 0].astype(int), neg_rw[:, 1:].astype(int)

        h_start = self.embedding[start].reshape(neg_rw.shape[0], 1, self.embedding_dim)
        h_rest = self.embedding[rest.reshape(-1)].reshape(neg_rw.shape[0], -1, self.embedding_dim)

        out = (h_start * h_rest).sum(axis=-1).reshape(-1)
        neg_loss = -jnp.log(1 - jax.nn.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss
