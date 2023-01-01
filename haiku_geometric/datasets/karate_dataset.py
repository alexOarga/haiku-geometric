# This code was obtained from:
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/karate.html#KarateClub
#  AND
# https://github.com/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb

import jraph
import jax.numpy as jnp
from haiku_geometric.datasets.base import DataGraphTuple, GraphDataset
from typing import List, Tuple


class KarateClub(GraphDataset):
    r"""TZachary's karate club network from the `"An Information Flow Model for
    Conflict and Fission in Small Groups"
    <http://www1.ind.ku.dk/complexLearning/zachary1977.pdf>`_ paper
    
    **Attributes:**
        - **data** List[DataGraphTuple]: List of length 1 containing a single graph.
    
    Stats:
        .. list-table::
            :widths: 10 10 10 10 10 10
            :header-rows: 1

            * - #nodes
              - #edges
              - #features
              - nodes features size
              - edge features size
              - #classes
            * - 34
              - 156
              - 34
              - 34
              - 0
              - 4
    """
    
    def __init__(self):
        """"""
        social_graph = [
                (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
                (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
                (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
                (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
                (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
                (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
                (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
                (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
                (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
                (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
                (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
                (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
                (33, 31), (33, 32)]
        # Add reverse edges.
        social_graph += [(edge[1], edge[0]) for edge in social_graph]
        n_club_members = 34

        y = jnp.asarray([  # Create communities.
            1, 1, 1, 1, 3, 3, 3, 1, 0, 1, 3, 1, 1, 1, 0, 0, 3, 1, 0, 1, 0, 1,
            0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0
        ])
        
        # Select a single training node for each community
        # (we just use the first one).
        train_mask = jnp.zeros(y.shape[0])
        for i in range(int(y.max()) + 1):
            train_mask.at[(y == i).nonzero()[0]].set(True)
            
        graph = DataGraphTuple(
            # One-hot encoding for nodes, i.e. argmax(nodes) = node index.
            nodes=jnp.eye(n_club_members),
            edges=None,
            senders=jnp.asarray([edge[0] for edge in social_graph]),
            receivers=jnp.asarray([edge[1] for edge in social_graph]),
            n_node=jnp.asarray([n_club_members]),
            n_edge=jnp.asarray([len(social_graph)]),
            globals=None,
            y = y,
            position=None,
            train_mask=train_mask
        )

        super().__init__([graph])
        
