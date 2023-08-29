import jax.numpy as jnp


def pad_graph(data_graph, n_nodes, n_edges, batch=None):
    r"""Pads the given graphs until they have the given number of nodes and edges.
    New nodes and edges are added in a new batch.

        Args:
            data_graph (:class:`haiku_geometric.datasets.base.DataGraphTuple`.): The graph to pad.
            n_nodes (int): The number of nodes to pad to.
            n_edges (int): The number of edges to pad to.
            batch (int, optional): Batch indexes of the given graph. It will be updated to include the new batch.

        Returns:
            - A single :class:`haiku_geometric.datasets.base.DataGraphTuple` containing the padded graph.
            - A `jax.numpy.ndarray` with boolean values indicating which nodes are old (True) and which are new (False).
            - A `jax.numpy.ndarray` with the new batch indexes.
            - A 'int' with the new number of batches (i.e. the old number of batches + 1).
        """
    num_nodes = int(jnp.sum(data_graph.n_node))
    num_edges = int(jnp.sum(data_graph.n_edge))
    pad_n_nodes = max(int(n_nodes - num_nodes), 0)
    pad_n_edges = max(int(n_edges - num_edges), 0)

    if data_graph.nodes is not None and pad_n_nodes > 0:
        shape = list(data_graph.nodes.shape)
        shape[0] = pad_n_nodes
        new_nodes = jnp.zeros(shape, dtype=data_graph.nodes[0].dtype)
        new_nodes = jnp.concatenate([data_graph.nodes, new_nodes], axis=0)
        data_graph = data_graph._replace(nodes=new_nodes)

    if pad_n_edges > 0:
        new_senders = jnp.zeros((pad_n_edges,), dtype=data_graph.senders[0].dtype)
        new_receivers = jnp.zeros((pad_n_edges,), dtype=data_graph.senders[0].dtype)
        new_senders = jnp.concatenate([data_graph.senders, new_senders], axis=0)
        new_receivers = jnp.concatenate([data_graph.receivers, new_receivers], axis=0)

        if data_graph.edges is not None:
            shape = list(data_graph.edges.shape)
            shape[0] = pad_n_edges
            new_edges = jnp.zeros(shape, dtype=data_graph.nodes[0].dtype)
            new_edges = jnp.concatenate([data_graph.edges, new_edges], axis=0)
            data_graph = data_graph._replace(edges=new_edges)
        data_graph = data_graph._replace(senders=new_senders)
        data_graph = data_graph._replace(receivers=new_receivers)

    new_n_node = jnp.concatenate([data_graph.n_node, jnp.array([pad_n_nodes])], axis=0)
    new_n_edge = jnp.concatenate([data_graph.n_edge, jnp.array([pad_n_edges])], axis=0)

    data_graph = data_graph._replace(n_node=new_n_node)
    data_graph = data_graph._replace(n_edge=new_n_edge)

    if batch is not None:
        num_batch = jnp.max(batch) + 1
        new_batch = jnp.full((pad_n_nodes,), num_batch)
        batch = jnp.concatenate([batch, new_batch], axis=0)

    if batch is not None:
        num_batch = jnp.max(batch) + 1
        mask = jnp.full((num_batch,), True)
        mask = mask.at[-1].set(False)
    else:
        mask = jnp.array([True, False])

    return data_graph, mask, batch, num_batch.item()