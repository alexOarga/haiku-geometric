import pytest
import haiku as hk
import jax

from haiku_geometric.datasets.toy_dataset import ToyGraphDataset


def test_toy_dataset():
    kc = ToyGraphDataset()
    assert kc.data[0].nodes.shape == (4, 1)
    assert kc.data[0].receivers.shape == (5,)
    assert kc.data[0].senders.shape == (5,)
    