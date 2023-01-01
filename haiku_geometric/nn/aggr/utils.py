from typing import Optional, Union, List
from haiku_geometric.nn.aggr import Aggregation, MaxAggregation, \
    SumAggregation, MeanAggregation, MinAggregation, MultiAggregation


def aggregation(aggr: Union[str, List[str]], *args, **kwargs) -> Aggregation:
    
    if isinstance(aggr, list):
        return MultiAggregation(aggr, *args, **kwargs)
    
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