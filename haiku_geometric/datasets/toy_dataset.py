# This code was obtained from:
# https://github.com/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb

import jraph
import jax.numpy as jnp
from haiku_geometric.datasets.base import DataGraphTuple, GraphDataset


class ToyGraphDataset(GraphDataset):
    r"""Toy dataset from `Deepmind's intro to graph nets tutorial with jraph
    <https://github.com/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb>`_.

    **Attributes:**
        - **data**: (List[DataGraphTuple]): List of graph tuples containing only one graph.

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10 10
            :header-rows: 1

            * - #nodes
              - #edges
              - #node features
              - #edge features
              - #global features
              - #classes
            * - 4
              - 5
              - 1
              - 1
              - 1
              - 0
    """
    
    def __init__(self):
        """"""
        # Nodes are defined implicitly by their features.
        # We will add four nodes, each with a feature, e.g.
        # node 0 has feature [0.],
        # node 1 has featre [2.] etc.
        # len(node_features) is the number of nodes.
        nodes = jnp.array([[0.], [2.], [4.], [6.]])

        # We will now specify 5 directed edges connecting the nodes we defined above.
        # We define this with `senders` (source node indices) and `receivers`
        # (destination node indices).
        # For example, to add an edge from node 0 to node 1, we append 0 to senders,
        # and 1 to receivers.
        # We can do the same for all 5 edges:
        # 0 -> 1
        # 1 -> 2
        # 2 -> 0
        # 3 -> 0
        # 0 -> 3
        senders = jnp.array([0, 1, 2, 3, 0])
        receivers = jnp.array([1, 2, 0, 0, 3])

        # You can optionally add edge attributes to the 5 edges.
        edges = jnp.array([[5.], [6.], [7.], [8.], [8.]])

        # We then save the number of nodes and the number of edges.
        # This information is used to make running GNNs over multiple graphs
        # in a GraphsTuple possible.
        n_node = jnp.array([4])
        n_edge = jnp.array([5])

        # Optionally you can add `global` information, such as a graph label.
        globals = jnp.array([[1]])  # Same feature dims as nodes and edges.

        graph = DataGraphTuple(
            nodes=nodes,
            edges=edges,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
            globals=globals,
            position=None,
            y=None,
            train_mask=None,
        )

        super().__init__([graph])