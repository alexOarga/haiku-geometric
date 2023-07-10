import jax.numpy as jnp
from typing import Any, NamedTuple, Iterable, Mapping, Union, Optional, List
ArrayTree = Union[jnp.ndarray, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]


class DataGraphTuple(NamedTuple):
    r""" Container class containing an individual graph data.
    
    Attributes:
    
    - nodes: If available, Array of node features.
    - edges: If available, Array of edge features.
    - receivers: Array of receiver node indices.
    - senders: Array of sender node indices.
    - globals: If available, array of global features.
    - n_node: Number of nodes in the graph.
    - n_edge: Number of edges in the graph.
    - y: If available, ground truth for each node, edge or whole graph.
    - position: If available, Array of node positions.
    - train_mask: If available, array of booleans indicating which elements are in the train set.
    
    """
    nodes: Optional[ArrayTree]
    edges: Optional[ArrayTree]
    receivers: Optional[jnp.ndarray]  # with integer dtype
    senders: Optional[jnp.ndarray]  # with integer dtype
    n_node: jnp.ndarray  # with integer dtype
    n_edge: jnp.ndarray   # with integer dtype
    globals: Optional[ArrayTree]
    position: Optional[jnp.ndarray]
    y: Optional[jnp.ndarray]
    train_mask: Optional[jnp.ndarray]


class GraphDataset():
    r""" Container class containing a list of graphs.

    Attributes:
    - data: List of DataGraphTuple.
    - y: If available, ground truth for each graph.
    """
    def __init__(self,
            data: List[DataGraphTuple] = [],
            y: Optional[jnp.ndarray] = None
        ):
        """"""
        self.data = data
        self.y = y
        
    def __repr__(self) -> str: 
        return str(self.__dict__)
        