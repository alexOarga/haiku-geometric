import haiku as hk
import jax.numpy as jnp
from typing import Optional, Union, List, Dict, Any
from jraph._src.utils import segment_sum, segment_mean, segment_max, segment_min_or_constant

from haiku_geometric.nn.aggr.base import Aggregation
from haiku_geometric.nn.aggr.utils import aggregation
from haiku_geometric.nn.aggr import MultiAggregation
from haiku_geometric.utils import degree

# Completely adapted from:
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/aggr/scaler.html#DegreeScalerAggregation
class DegreeScalerAggregation(Aggregation):
    r"""
    Combines ,multiple aggregation operators and transforms its outputs with scalers
    as described in the: `"Principal Neighbourhood Aggregation for
    Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper.
    This aggregation is used in the :obj:`PNAConv` convolution layer.
    
    Args:
        aggr (string or list or Aggregation): Aggregation or list of aggregation
            operators to be used.
        scaler (str or list): List of scaling function identifiers. Available 
            scalers are: :obj:`"identity"`, :obj:`"amplification"`,
            :obj:`"attenuation"`, :obj:`"linear"` and :obj:`"inverse_linear"`.
        deg (jnp.ndarray): Histogram of in-degrees of nodes in the training set,
            used by scalers to normalize.
        train_norm (bool, optional) If :obj:`True`, normalization parameters are learned.
            (default: :obj:`False`)
    
    """
    def __init__(
        self,
        aggr: Union[str, List[str], Aggregation],
        scaler: Union[str, List[str]],
        deg: jnp.ndarray,
        train_norm: bool = False):
        
        super().__init__()
        
        if isinstance(aggr, str) or isinstance(aggr, list):
            self.aggr = aggregation(aggr)
        else:
            self.aggr = aggr
        
        self.scaler = [scaler] if isinstance(scaler, str) else scaler
        N = jnp.sum(deg).astype(int)
        bin_degree = jnp.arange(deg.size)
        
        self.init_avg_deg_lin = (bin_degree * deg).sum().astype(float) / N
        self.init_avg_deg_log = ((jnp.log(bin_degree + 1) * deg).sum()).astype(float) / N
        
        if train_norm:
            self.avg_deg_lin = hk.get_parameter("avg_deg_lin", shape=[1, 1], init=hk.initializers.RandomNormal())
            self.avg_deg_log = hk.get_parameter("avg_deg_log", shape=[1, 1], init=hk.initializers.RandomNormal())
        else:
            self.avg_deg_lin = self.init_avg_deg_lin
            self.avg_deg_log = self.init_avg_deg_log
            
    def __call__(
        self,
        data: jnp.ndarray,
        receivers: jnp.ndarray, 
        num_segments: Optional[int] = None,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        dim: int = 0):
        
        out = self.aggr(data, receivers, num_segments, \
                        indices_are_sorted, unique_indices)
            
        deg = degree(receivers, num_segments)
        size = [1] * len(out.shape)
        size[dim] = -1
        deg = deg.reshape(size)
        
        outs = []
        for scaler in self.scaler:
            if scaler == 'identity':
                out_scaler = out
            elif scaler == 'amplification':
                out_scaler = out * (jnp.log(deg + 1) / self.avg_deg_log)
            elif scaler == 'attenuation':
                out_scaler = out * (self.avg_deg_log / jnp.log(deg + 1))
            elif scaler == 'linear':
                out_scaler = out * (deg / self.avg_deg_lin)
            elif scaler == 'inverse_linear':
                out_scaler = out * (self.avg_deg_lin / deg)
            else:
                raise ValueError(f"Unknown scaler '{scaler}'")
            outs.append(out_scaler)

        return jnp.concatenate(outs, axis=-1) if len(outs) > 1 else outs[0]