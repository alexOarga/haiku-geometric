from haiku_geometric.nn.aggr.base import *
from haiku_geometric.nn.aggr.basic import *
from haiku_geometric.nn.aggr.multi import *
from haiku_geometric.nn.aggr.degree_scaler import *


__all__ = [
    'Aggregation',
    'SumAggregation',
    'MeanAggregation',
    'MaxAggregation',
    'MinAggregation',
    'MultiAggregation',
    'DegreeScalerAggregation',
]

classes = __all__