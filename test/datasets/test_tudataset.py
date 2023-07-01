import pytest
import haiku as hk
import jax

from haiku_geometric.datasets.tu_dataset import TUDataset


def test_toy_dataset():
    dtt = TUDataset(root='/tmp/MUTAG', name='MUTAG', use_node_attr=True, use_edge_attr=True)
    assert dtt.data[0].nodes.shape[-1] == 7
    assert dtt.data[0].edges.shape[-1] == 4
    assert len(dtt.data) == 188
    