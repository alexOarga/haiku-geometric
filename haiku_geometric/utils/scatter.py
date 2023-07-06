import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial


# TODO: swapaxes depends on dim parameter, which makes this function partially jit-able
@partial(jax.jit, static_argnums=(1,))
def scatter(input, dim, index, src, reduce=None):
    # This is AranKomat's (https://github.com/AranKomat) implementation of scatter
    # It is intented to work as pytorch's scatter function.
    dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,),
                                            scatter_dims_to_operand_dims=(0,))

    if reduce is None:
        _scatter = jax.lax.scatter
    elif reduce == "add":
        _scatter = jax.lax.scatter_add
    elif reduce == "multiply":
        _scatter = jax.lax.scatter_mul

    _scatter = partial(_scatter, dimension_numbers=dnums)
    vmap_inner = partial(vmap, in_axes=(0, 0, 0), out_axes=0)

    for _ in range(len(input.shape) - 1):
        _scatter = vmap_inner(_scatter)
    swap = lambda x: jnp.swapaxes(x, dim, -1)
    input, index, src = list(map(swap, (input, index, src)))
    return swap(_scatter(input, jnp.expand_dims(index, axis=-1), src))