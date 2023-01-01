import haiku as hk
import jraph
import jax.numpy as jnp
from typing import Optional, Union


class Linear(hk.Module):
    r"""Applies a linear module on the input features

    .. math::
        \mathbf{x}^{\prime} = \mathbf{x} \mathbf{W}^{\top} + \mathbf{b}


    Args:
        out_channels (int): Size of each output features.
        bias (bool, optional): Whether to add a bias to the output. (default: :obj:`True`)
        weight_initializer: Optional initializer for weights. By default, uses random values from truncated normal,
            with stddev 1 / sqrt(fan_in).
        bias_initializer: Optional initializer for the bias. Default to zeros. (default: :obj:`None`)
    """

    def __init__(
            self,
            out_channels: int,
            bias: bool = True,
            weight_initializer: hk.initializers.Initializer = None,
            bias_initializer: hk.initializers.Initializer = None):
        """"""
        super().__init__()
        # We use the already defined haiku layer
        self.linear = hk.Linear(
            out_channels,
            with_bias=bias,
            w_init=weight_initializer,
            b_init=bias_initializer
        )

    def __call__(
            self,
            x: jnp.ndarray = None,
            graph: jraph.GraphsTuple = None
    ) -> Union[jnp.ndarray, jraph.GraphsTuple]:
        """"""
        # This function is just a haiku Linear applied to the nodes of a GrpahTuple
        if graph is not None:
            nodes = graph.nodes
        else:
            nodes = x

        nodes = self.linear(nodes)

        if graph is not None:
            graph = graph._replace(nodes=nodes)
            return graph
        else:
            return nodes