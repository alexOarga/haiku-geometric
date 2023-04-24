import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from typing import Optional, Tuple
from jax.experimental.sparse import BCOO
from haiku_geometric.utils import random_walk
from haiku_geometric.utils import num_nodes as _num_nodes


class Node2Vec(hk.Module):
    """
    The Node2vec model from the paper:
    `"node2vec: Scalable Feature Learning for Networks" <https://arxiv.org/abs/1607.00653>`_ paper.

    Args:
        senders (jnp.ndarray): The source nodes of the graph.
        receivers (jnp.ndarray): The target nodes of the graph.
        embedding_dim (int): The dimensionality of the node embeddings.
        walk_length (int): The length of the random walk.
        context_size (int): Context size considered for positive sampling.
        walks_per_node (int, optional): Number of walks per node. (default: :obj:`1`)
        p (float, optional): Likelihood of revisiting a node in the walk. (default: :obj:`1.0`).
        q (float, optional): Control parameter to interpolate between breadth-first strategy and depth-first strategy. (default: :obj:`1.0`).
        num_negative_samples (int, optional): Number of negative samples for each positive sample. (default: :obj:`1`).
        num_nodes (int, optional): The number of nodes in the graph. (default: :obj:`None`).
        rng (jax.random.PRNGKey, optional): The random number generator seed. (default: :obj:`jax.random.PRNGKey(42)`).

    **Attributes:**

        - **embedding** (jnp.ndarray): Embeddings of the node2vec model.
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
            rng=jax.random.PRNGKey(42),
         ):
        """"""
        super().__init__()

        N = _num_nodes(senders, receivers, num_nodes)
        self.num_nodes = N
        self.senders = senders
        self.receivers = receivers
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

        w_init = hk.initializers.TruncatedNormal()
        self.embedding = hk.get_parameter("embedding", shape=[self.num_nodes, self.embedding_dim], init=w_init)


    def pos_sample(self, batch: jnp.ndarray):
        """Returns positive samples."""
        #batch = jnp.repeat(batch, self.walks_per_node) # TODO: is batch needed?
        rw = random_walk(
            np.asarray(self.senders),
            np.asarray(self.receivers),
            self.walk_length,
            self.p,
            self.q)
        rw = jnp.array(rw)
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw - 1):
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

    def __call__(self):
        """ This is the loss function of the node2vec model.

        Returns:
            - Current embeddings of the model (jnp.ndarray).
            - Loss computed in this forward call.
        """
        pos_rw, neg_rw = self.sample(jnp.arange(self.num_nodes))
        start, rest = pos_rw[:, 0].astype(int), pos_rw[:, 1:].astype(int)

        h_start = self.embedding[start].reshape(pos_rw.shape[0], 1, self.embedding_dim)
        h_rest = self.embedding[rest.reshape(-1)].reshape(pos_rw.shape[0], -1, self.embedding_dim)

        out = (h_start * h_rest).sum(axis=-1).reshape(-1)
        pos_loss = -jnp.log(jax.nn.sigmoid(out) + self.EPS).mean()

        start, rest = neg_rw[:, 0].astype(int), neg_rw[:, 1:].astype(int)

        h_start = self.embedding[start].reshape(neg_rw.shape[0], 1, self.embedding_dim)
        h_rest = self.embedding[rest.reshape(-1)].reshape(neg_rw.shape[0], -1, self.embedding_dim)

        out = (h_start * h_rest).sum(axis=-1).reshape(-1)
        neg_loss = -jnp.log(1 - jax.nn.sigmoid(out) + self.EPS).mean()

        return self.embedding, pos_loss + neg_loss
