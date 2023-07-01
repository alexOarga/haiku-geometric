from .toy_dataset import ToyGraphDataset
from .karate_dataset import KarateClub
from .ogb_dataset import OGB
from .planetoid_dataset import Planetoid
from .gnn_benchmark_dataset import GNNBenchmarkDataset
from .tu_dataset import TUDataset
from .base import GraphDataset, DataGraphTuple

__all__ = [
    'GraphDataset',
    'DataGraphTuple',
    'ToyGraphDataset',
    'KarateClub',
    'OGB',
    'Planetoid',
    'GNNBenchmarkDataset',
    'TUDataset'
]

classes = __all__