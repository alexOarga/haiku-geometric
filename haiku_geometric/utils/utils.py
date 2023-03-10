import jax.numpy as jnp
from jraph._src.utils import segment_sum
from typing import Optional

def degree(index: jnp.ndarray, total_num_nodes):
    d = jnp.ones_like(index)
    return segment_sum(
        d, index, total_num_nodes)

def num_nodes(senders: jnp.ndarray, receivers: jnp.ndarray, num_nodes: Optional[int] = None):
    if num_nodes is None:
        return jnp.max(jnp.concatenate([senders, receivers])) + 1
    else:
        return num_nodes