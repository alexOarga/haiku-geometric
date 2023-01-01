import haiku as hk
import jax
import jax.numpy as jnp

class Aggregation(hk.Module):
    def __init__(self):
        super().__init__()
    def __call__(
            self,
            data: jnp.ndarray,
            segment_ids: jnp.ndarray):
        pass