import jax.numpy as jnp
import jax
import haiku as hk
from functools import partial
from typing import Optional, Union
from haiku_geometric.utils import scatter, batch_softmax

MIN_INF = -65504.0


class TopKPooling(hk.Module):
    r""" Topk pooling operator from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ and `"Towards Sparse Hierarchical Graph Classifiers"
    <https://arxiv.org/abs/1811.01287>`_ paper.

    Args:
        in_channels (int): Dimension of input node features.
        ratio: (Union[int, float], optional): Ratio of nodes to keep.
            If int, the number of nodes to keep.
            (default: :obj:`0.5`).
        multiplier (float, optional): Multiplier to scale the features after pooling.
            (default: :obj:`1.`).
    """
    def __init__(self,
                 in_channels: int,
                 ratio: Union[int, float] = 0.5,
                 multiplier: float = 1.,
                 ):
        """"""
        super().__init__()
        p_init = hk.initializers.TruncatedNormal() #w_init = hk.initializers.TruncatedNormal(1. / jnp.sqrt(j)) # TODO: initialize with 1/sqrt(j)
        self.p = hk.get_parameter("p", shape=[in_channels,], init=p_init)
        self.ratio = ratio
        self.multiplier = multiplier


    def __call__(self,
                 x: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 edges: Optional[jnp.ndarray] = None,
                 batch: Optional[jnp.ndarray] = None,
                 create_new_batch: bool = False,
                 batch_size: int = None,
                 max_num_nodes: int = None,
                 ):
        r"""
        Args:
            x (jnp.ndarray): Node features of shape :obj:`[num_nodes, in_channels]`.
            senders (jnp.ndarray): Sender indices.
            receivers (jnp.ndarray): Receiver indices.
            edges (jnp.ndarray, optional): Edge features of shape :obj:`[num_edges, in_channels]`.
                (default: :obj:`None`).
            batch (jnp.ndarray, optional): Batch array with batch indexes for each node. Shape: :obj:`[num_nodes]`.
                **Note:** This array should be sorted in increasing order.
                (default: :obj:`None`).
            create_new_batch (bool, optional): If set to :obj:`False`, nodes that are not top-k selected and their edges
                are removed from the graph. If set to :obj:`True`, the nodes are kept in the graph, but they are assigned
                to a new batch with value :obj:`batch_size + 1`. Their corresponding edges are transformed in self-loops.
                **Note:** If :obj:`True`, the output sizes of :obj:`x`, :obj:`batch`, :obj:`senders`, :obj:`receivers` and :obj:`edges`
                stay the same. If :obj:`False` output sizes of :obj:`x`, :obj:`batch`, :obj:`senders`, :obj:`receivers` and :obj:`edges`
                might be reduced according to the :obj:`ratio` parameter.
                (default: :obj:`False`).
            batch_size (int, optional): Number of batched graphs. If not given, it is automatically computed as :obj:`batch.max() + 1`.
                (default: :obj:`None`).
            max_num_nodes (int, optional): Maximum number of nodes that a graph can have. If not given, it is automatically computed as
                :obj:`batch.shape[0]`.
                (default: :obj:`None`).

        Returns:
            :obj:`Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]`:
            - The updated nodes features.
            - The updated senders indices.
            - The updated receivers indices.
            - The updated edges features.
            - The updated batch array.

        **Observations:**
            To make this layer jit-able, it requires providing parameters :obj:`create_new_batch=True` and :obj:`batch_size`
            as static parameters.
        """
        num_nodes = x.shape[0]
        if batch is None:
            batch = jnp.zeros((num_nodes,), dtype=jnp.int32)
        x = x.reshape((-1, 1)) if x.ndim == 1 else x
        score = x * self.p
        score = jnp.sum(score, axis=-1)
        score = batch_softmax(score, batch, batch_size)

        if create_new_batch:
            return self._select_and_batch_topk(x, senders, receivers, edges, batch, score, batch_size)
        else:
            return self._select_topk(x, senders, receivers, edges, batch, score, num_nodes)


    def _select_and_batch_topk(self, x, senders, receivers, edges, batch, score, batch_size):
        new_batch, perm = select_batch_topk(score, self.ratio, batch, batch_size)

        x = x[perm]
        score = score[perm]

        x = x * score.reshape(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        new_batch_idx = jnp.max(batch) + 1

        # assign self loops to new batch nodes
        mask = (new_batch[senders] == new_batch_idx) | (new_batch[receivers] == new_batch_idx)
        senders = jnp.where(mask, receivers, senders)
        receivers = jnp.where(mask, senders, receivers)
        # no need to change edge attr

        return x, senders, receivers, edges, new_batch

    def _select_topk(self, x, senders, receivers, edges, batch, score, num_nodes):
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
        edge_attr = edges[mask] if edges is not None else None

        # pool new graphs by creating batches
        if batch is not None:
            out = jnp.arange(perm.shape[0])
            batch = scatter(out, 0, cluster_index, batch[perm])

        return x, senders, receivers, edge_attr, batch


@partial(jax.jit, static_argnums=(1))
def _count_occurrences(arr, rrange):
    '''
    Given an array of positive integers, sequentially count the number of occurrences of
    each integer (starting at 0).
    For instance, given the array [0, 1, 1, 2, 2, 2, 0, 1, 1, 1], the output
    would be [0, 0, 1, 0, 1, 2, 2, 2, 3, 4].
    Given the array [1, 2, 3, 4, 5, 1, 1]
    the output would be [0, 0, 0, 0, 0, 1, 2]

    Parameters:
        arr (jnp.ndarray): Array of positive integers.
        rrange (int): Range of the integers in the array.
            It will be iterated similar to the 'range(rrange)' function.
    '''

    counts = jnp.zeros_like(arr)

    def body_fun(val, counts):
        mask = (arr == val)
        count = jnp.cumsum(mask)
        counts = jnp.where(mask, count, counts)
        return counts

    counts = jax.lax.fori_loop(0, rrange, body_fun, counts)

    return counts

def select_batch_topk(score, ratio, batch, batch_size=None):

    if batch_size is None:
        batch_size = jnp.max(batch) + 1

    nodes_per_batch = jax.ops.segment_sum(
        data=jnp.ones(score.shape[0], dtype=jnp.int32),
        segment_ids=batch, num_segments=batch_size)

    perm = jnp.argsort(-score, axis=-1)  # trick for descending order
    # Sort batch according to the score
    # This arrays will be cropped later according to the ratio param.
    batch = batch[perm]
    occ = _count_occurrences(batch, batch_size) - 1

    if ratio >= 1:
        k = jnp.full((batch_size,), int(ratio))
        k = jnp.minimum(k, nodes_per_batch)
    else:
        k = jnp.ceil(ratio * nodes_per_batch).astype(jnp.int32)

    threshold_per_batch = k[batch] - 1
    new_batch = jnp.where(occ <= threshold_per_batch, batch, jnp.full(batch.shape, batch_size))
    batch_size += 1
    return new_batch, perm


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
    perm = jnp.argsort(-score_batch_matrix, axis=-1)  # trick for descending order
    if ratio >= 1:
        k = jnp.full((num_batch,), int(ratio))
        k = jnp.minimum(k, nodes_per_batch)
    else:
        k = jnp.ceil(ratio * nodes_per_batch).astype(jnp.int32)

    perm = perm + cum_num_nodes[:, None]
    index = [(b, i) for b in range(num_batch) for i in range(k[b])]
    index = tuple(jnp.array(index).T)
    perm = perm[index]
    perm = perm.reshape(-1)
    return perm





