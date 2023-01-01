from .toy_dataset import ToyGraphDataset
from .karate_dataset import KarateClub
from .ogb_dataset import OGB
from .planetoid_dataset import Planetoid
from .gnn_benchmark_dataset import GNNBenchmarkDataset

__all__ = [
    'ToyGraphDataset',
    'KarateClub',
    'OGB',
    'Planetoid',
    'GNNBenchmarkDataset',
]

classes = __all__