from .utils import *
from .laplacian import get_laplacian, get_laplacian_matrix, eigv_laplacian
from .random_walk import random_walk
from .coalesce import coalesce
from .undirected import to_undirected
from .magnetic_laplacian import eigv_magnetic_laplacian

__all__ = [
    'coalesce',
    'get_laplacian',
    'get_laplacian_matrix',
    'eigv_laplacian',
    'eigv_magnetic_laplacian',
    'random_walk',
    'to_undirected',
    'fill_diagonal'
]


classes = __all__