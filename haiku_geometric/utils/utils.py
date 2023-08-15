import jax
import jax.numpy as jnp
from jraph._src.utils import segment_sum
from typing import Optional
from functools import partial


def degree(index: jnp.ndarray, total_num_nodes):
    d = jnp.ones_like(index)
    return segment_sum(
        d, index, total_num_nodes)

def num_nodes(senders: jnp.ndarray, receivers: jnp.ndarray, num_nodes: Optional[int] = None):
    if num_nodes is None:
        return jnp.max(jnp.concatenate([senders, receivers])) + 1
    else:
        return num_nodes

# Function not yet avilable in JAX numpy
# See: https://github.com/google/jax/issues/2680
def fill_diagonal(a, val):
  assert a.ndim >= 2
  i, j = jnp.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)


def batch_softmax(x: jnp.ndarray,
                batch: jnp.ndarray,
                num_segments):
    r"""Batched soft max."""
    batch_max = jax.ops.segment_max(
        data=jax.lax.stop_gradient(x),
        segment_ids=batch,
    num_segments=num_segments)
    batch_max = batch_max[batch]
    x = jnp.exp(x - batch_max)
    batch_sum = jax.ops.segment_sum(
        data=x,
        segment_ids=batch,
    num_segments=num_segments) + 1e-16
    batch_sum = batch_sum[batch]
    return x / batch_sum