import jax.numpy as jnp


def global_add_pool(x: jnp.ndarray) -> jnp.ndarray:
    r"""Returns the sum of all node features of the input graph:

    .. math::
        \mathbf{r} = \sum_{i=1}^{N} \mathbf{h}_i.

    Args:
        x (jax.numpy.ndarray): Node features array.

    Returns;
        (jax.numpy.ndarray): Array with the sum of the nodes features.

    """
    return jnp.sum(x, axis=-2, keepdims=True)


def global_mean_pool(x: jnp.ndarray) -> jnp.ndarray:
    r"""Returns the average of all node features of the input graph:

    .. math::
        \mathbf{r} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{h}_i.

    Args:
        x (jax.numpy.ndarray): Node features array.

    Returns;
        (jax.numpy.ndarray): Array with the average of the nodes features.

    """
    return jnp.mean(x, axis=-2, keepdims=True)


def global_max_pool(x: jnp.ndarray) -> jnp.ndarray:
    r"""Returns the maximum across the input features.
    The maximum is performed individually over each channel.

    .. math::
        \mathbf{r} = \max_{i=1}^{N} \mathbf{h}_i.

    Args:
        x (jax.numpy.ndarray): Node features array.

    Returns;
        (jax.numpy.ndarray): Array with the average of the nodes features.

    """
    return jnp.max(x, axis=-2, keepdims=True)