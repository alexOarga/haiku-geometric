import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial


def _scatter(input, dim, index, src, reduce):
    dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,),
                                            scatter_dims_to_operand_dims=(0,))

    _scatterf = partial(reduce, dimension_numbers=dnums)
    vmap_inner = partial(vmap, in_axes=(0, 0, 0), out_axes=0)

    for _ in range(len(input.shape) - 1):
        _scatterf = vmap_inner(_scatterf)
    def swap(x):
        if x.ndim == 1:
            return x
        return jnp.swapaxes(x, dim, -1)
    #swap = lambda x: jnp.swapaxes(x, dim, -1)
    input, index, src = list(map(swap, (input, index, src)))
    return swap(_scatterf(input, jnp.expand_dims(index, axis=-1), src))


def scatter(input, dim, index, src, reduce=None):
    # This is AranKomat's (https://github.com/AranKomat) implementation of scatter
    # It is intented to work as pytorch's scatter function.

    if reduce is None:
        _reduce = jax.lax.scatter
    elif reduce == "sum":
        _reduce = jax.lax.scatter_add
    elif reduce == "multiply":
        _reduce = jax.lax.scatter_mul
    elif reduce == "max":
        _reduce = jax.lax.scatter_max

    return _scatter(input, dim, index, src, _reduce)
