import pytest
import haiku as hk
import jax

from haiku_geometric.datasets.karate_dataset import KarateClub


def test_karate():
    kc = KarateClub()
    assert kc.data[0].nodes.shape == (34, 34)
    assert kc.data[0].receivers.shape == (156,)
    assert kc.data[0].senders.shape == (156,)
    