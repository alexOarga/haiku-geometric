import haiku as hk
import jax.numpy as jnp
from typing import Optional, Union, List, Dict, Any
from jraph._src.utils import segment_sum, segment_mean, segment_max, segment_min_or_constant

from haiku_geometric.nn.aggr import Aggregation, MaxAggregation, \
    SumAggregation, MeanAggregation, MinAggregation


#: TODO: merge this function into haiku_geometric/nn/aggr/utils.py
def simple_aggregation_resolver(aggr: str) -> Aggregation:
    if aggr == 'max':
        return MaxAggregation()
    elif aggr == 'min':
        return MinAggregation()
    elif aggr == 'sum' or aggr == 'add':
        return SumAggregation()
    elif aggr == 'mean':
        return MeanAggregation()
    else:
        raise ValueError(f"Aggregation operator {aggr} is not defined.")


class MultiAggregation(Aggregation):
    """
    Performs the aggregation of multiple aggregators operators.
    The aggregation is performed according to the :obj:`mode` parameter.
    
    Args:
        aggrs (list): List of aggregation operators.
        aggrs_kwargs (list, optional): Optional arguments passed to the aggregator operator
            function. This must be a list of dictionaries of length equal to the length of 
            :obj:`aggrs` parameter.
            (default: :obj:`None`)
        mode (string, optional): The combine mode used to aggregate the aggregation operators
            result. Available modes are:
            :obj:`"cat"`, obj:`"proj"`, :obj:`"sum"`, :obj:`"mean"`, 
            :obj:`"max"`, :obj:`"min"`, :obj:`"std"`, 
            :obj:`"var"`. 
            (default: :obj:`"cat"`)
        mode_kwargs (dict, optional): Optimal arguments passed to the mode function.
            (default: :obj:`None`)
        
    """

    #: TODO: Add support for "logsumexp" and attention modes.
    def __init__(
            self,
            aggrs: List[Union[Aggregation, str]],
            aggrs_kwargs: Optional[List[Dict[str, Any]]] = None,
            mode: Optional[str] = 'cat',
            mode_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super().__init__()

        if not isinstance(aggrs, (list, tuple)):
            raise ValueError(f"Aggregators of MultiAggregation must be a list or tuple")

        if aggrs_kwargs is None:
            aggrs_kwargs = [{}] * len(aggrs)
        elif len(aggrs) != len(aggrs_kwargs):
            raise ValueError("In MultiAggregation, parameter 'aggrs_kwargs' must be "
                             "the same size as parameter 'aggrs'.")

        self.aggrs = [simple_aggregation_resolver(aggr, **aggr_kwargs) for aggr, aggr_kwargs in
                      zip(aggrs, aggrs_kwargs)]
        self.mode = mode

        if mode == 'proj':
            self.out_channels = mode_kwargs.pop('out_channels', None)
            if self.out_channels is None:
                raise ValueError("MultiAggregation with mode 'proj' requires the parameter"
                                 "'out_channels' specified")
            self.linear = hk.Linear(self.out_channels)

        dense_combine_modes = [
            'sum', 'mean', 'max', 'min', 'std', 'var'
        ]
        if mode in dense_combine_modes:
            self.dense_combine = getattr(jnp, mode)

    def __call__(
            self,
            data: jnp.ndarray,
            receivers: jnp.ndarray,
            num_segments: Optional[int] = None,
            indices_are_sorted: bool = False,
            unique_indices: bool = False):

        outs = [0] * len(self.aggrs)

        for i, aggr in enumerate(self.aggrs):
            outs[i] = aggr(data, receivers, num_segments, indices_are_sorted, unique_indices)

        if len(outs) == 1:
            return outs[0]

        if self.mode == 'cat':
            return jnp.concatenate(outs, axis=-1)

        if self.mode == 'proj':
            return self.linear(jnp.concatenate(outs, axis=-1))

        if hasattr(self, 'dense_combine'):
            out = self.dense_combine(jnp.stack(outs, axis=0), axis=0)

        raise ValueError(f"Combine mode '{self.mode}' is not supported.")
