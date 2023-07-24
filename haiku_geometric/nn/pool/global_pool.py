import jax
import jax.numpy as jnp


def global_add_pool(x: jnp.ndarray, batch: jnp.ndarray=None, num_segments=None) -> jnp.ndarray:
    r"""Returns the sum of all node features of the input graph:

    .. math::
        \mathbf{r} = \sum_{i=1}^{N} \mathbf{h}_i.

    Args:
        x (jax.numpy.ndarray): Node features array.
        batch (jax.numpy.ndarray, optional): Batch vector with indices that indicate to which graph each node belongs.
            (default: :obj:`None`).
        num_segments (int, optional): Number of segments in :obj:`batch`. (default: :obj:`None`)

    Returns:
        (jax.numpy.ndarray): Array with the sum of the nodes features. If :obj:`batch` is not :obj:`None`, the
        output array will have shape :obj:`[batch_size, *]`, where :obj:`*` denotes the remaining dimensions.

    """
    dim = -1 if x.ndim == 1 else -2
    if batch is None:
        return jnp.sum(x, axis=dim, keepdims=True)
    return jax.ops.segment_sum(x, batch, num_segments=num_segments)


def global_mean_pool(x: jnp.ndarray, batch: jnp.ndarray=None, num_segments=None) -> jnp.ndarray:
    r"""Returns the average of all node features of the input graph:

    .. math::
        \mathbf{r} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{h}_i.

    Args:
        x (jax.numpy.ndarray): Node features array.
        batch (jax.numpy.ndarray, optional): Batch vector with indices that indicate to which graph each node belongs.
            (default: :obj:`None`).
        num_segments (int, optional): Number of segments in :obj:`batch`. (default: :obj:`None`)

    Returns:
        (jax.numpy.ndarray): Array with the average of the nodes features. If :obj:`batch` is not :obj:`None`, the
        output array will have shape :obj:`[batch_size, *]`, where :obj:`*` denotes the remaining dimensions.

    """
    dim = -1 if x.ndim == 1 else -2
    if batch is None:
        return jnp.mean(x, axis=dim, keepdims=True)
    sum = jax.ops.segment_sum(x, batch, num_segments=num_segments)
    count = jax.ops.segment_sum(jnp.ones_like(x), batch, num_segments=num_segments)
    return sum / jnp.maximum(count, 1)


def global_max_pool(x: jnp.ndarray, batch: jnp.ndarray=None, num_segments=None) -> jnp.ndarray:
    r"""Returns the maximum across the input features.
    The maximum is performed individually over each channel.

    .. math::
        \mathbf{r} = \max_{i=1}^{N} \mathbf{h}_i.

    Args:
        x (jax.numpy.ndarray): Node features array.
        batch (jax.numpy.ndarray, optional): Batch vector with indices that indicate to which graph each node belongs.
            (default: :obj:`None`).
        num_segments (int, optional): Number of segments in :obj:`batch`. (default: :obj:`None`)

    Returns:
        (jax.numpy.ndarray): Array with the average of the nodes features. If :obj:`batch` is not :obj:`None`, the
        output array will have shape :obj:`[batch_size, *]`, where :obj:`*` denotes the remaining dimensions.

    """
    dim = -1 if x.ndim == 1 else -2
    if batch is None:
        return jnp.max(x, axis=dim, keepdims=True)
    return jax.ops.segment_max(x, batch, num_segments=num_segments)