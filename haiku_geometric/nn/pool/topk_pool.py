import jax.numpy as jnp
import jax
import haiku as hk
from typing import Optional, Union
from haiku_geometric.utils import scatter, batch_softmax

MIN_INF = -65504.0


def topk_indexes(score, ratio, batch):

    nodes_per_batch = jax.ops.segment_sum(
        data=jnp.ones(score.shape[0], dtype=jnp.int32),
        segment_ids=batch)

    cum_num_nodes = jnp.concatenate(
        [jnp.zeros(1),
         jnp.cumsum(nodes_per_batch, axis=0)[:-1]], axis=0, dtype=jnp.int32)

    num_batch = int(batch.max() + 1)
    max_num_nodes = jnp.max(nodes_per_batch)
    score_batch_matrix = jnp.full((num_batch, max_num_nodes), MIN_INF)
    index = [(b, i) for b in range(num_batch) for i in range(nodes_per_batch[b])]
    index = tuple(jnp.array(index).T)
    # TODO: compare performance with indexing a flat array
    score_batch_matrix = score_batch_matrix.at[index].set(score, unique_indices=True)
    perm = jnp.argsort(score_batch_matrix, axis=-1)[::-1]  # trick for descending order
    if ratio >= 1:
        k = jnp.full((num_batch,), int(ratio))
        k = jnp.minimum(k, nodes_per_batch)
    else:
        k = jnp.floor(ratio * nodes_per_batch).astype(jnp.int32)

    perm = perm + cum_num_nodes[:, None]
    index = [(b, i) for b in range(num_batch) for i in range(k[b])]
    index = tuple(jnp.array(index).T)
    perm = perm[index]
    perm = perm.reshape(-1)
    return perm


class TopKPooling(hk.Module):
    r"""
    """
    def __init__(self,
                 in_channels: int,
                 ratio: Union[int, float] = 0.5,
                 min_score: Optional[float] = None,
                 multiplier: float = 1.,
                 ):
        r"""Topk pooling operator."""
        #w_init = hk.initializers.TruncatedNormal(1. / jnp.sqrt(j)) # TODO: initialize with 1/sqrt(j)
        super().__init__()
        p_init = hk.initializers.TruncatedNormal()
        self.p = hk.get_parameter("p", shape=[in_channels,], init=p_init)
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier


    def __call__(self,
                 x: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 edge_attr: Optional[jnp.ndarray] = None,
                 batch: Optional[jnp.ndarray] = None
         ):
        """"""
        num_nodes = x.shape[0]
        if batch is None:
            batch = jnp.zeros((num_nodes,), dtype=jnp.int32)
        x = x.reshape((-1, 1)) if x.ndim == 1 else x
        score = x * self.p
        score = jnp.sum(score, axis=-1)
        score = batch_softmax(score, batch, int(jnp.max(batch) + 1))

        perm = topk_indexes(score, self.ratio, batch)
        score = score[perm]

        x = x[perm] * score.reshape(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        # filter adjacency matrix
        cluster_index = jnp.arange(perm.shape[0])
        mask = jnp.full((num_nodes,), -1)
        mask = mask.at[perm].set(cluster_index)
        senders, receivers = mask[senders], mask[receivers]
        mask = (senders >= 0) & (receivers >= 0)
        senders, receivers = senders[mask], receivers[mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None

        # pool new graphs by creating batches
        if batch is not None:
            batch = jnp.arange(perm.shape[0])
            batch = scatter(batch, 0, cluster_index, batch[perm])

        return x, senders, receivers, edge_attr, batch








