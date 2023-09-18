import os
import jax
import glob
import jax.numpy as jnp
from haiku_geometric.datasets.base import DataGraphTuple, GraphDataset
from haiku_geometric.datasets.utils import download_url, extract_zip
import haiku_geometric as hkg


class TUDataset(GraphDataset):
    r"""The TUDataset from the `"TUDataset: A collection of benchmark datasets for learning with graphs"
    <https://arxiv.org/abs/2007.08663>`_ paper.

    Parameters:
        name (str): Name of a `TUDataset <https://chrsmrrs.github.io/datasets/docs/datasets/>`_, e.g. ``'ENZYMES'``, ``'PROTEINS'``, ``'MUTAG'``.
        root (str): Root directory where the dataset will be saved.
        use_node_attr (bool): If ``True``, the node attributes will be included in the graphs. (default: ``False``)
        use_edge_attr (bool): If ``True``, the edge attributes will be included in the graphs. (default: ``False``)

    **Attributes:**
        - **data**: (List[DataGraphTuple]): List of graph tuples (in this case only one graph).
        - **y**: (jnp.ndarray): Graph labels.

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #graphs
              - #avg nodes
              - #avg edges
              - #node features
              - #edge features
              - #classes
            * - PROTEINS
              - 1113
              - 39.06
              - 72.82
              - 4
              - 0
              - 2
            * - ENZYMES
              - 600
              - 32.63
              - 62.14
              - 21
              - 0
              - 6
            * - MUTAG
              - 188
              - 17.93
              - 19.79
              - 7
              - 4
              - 2

    """

    def __init__(
            self, name: str,
            root: str,
            use_node_attr: bool = False,
            use_edge_attr: bool = False):
        graphs, y = read_tu_dataset(root, name)
        if not use_node_attr:
            for i in range(len(graphs)):
                graphs[i] = graphs[i]._replace(nodes=None)
        if not use_edge_attr:
            for i in range(len(graphs)):
                graphs[i] = graphs[i]._replace(edges=None)
        super().__init__(graphs, y)


def read_file(file):
    f = open(file, "r")
    res = []
    for x in f:
        res.append(x)
    return res


def read_tu_dataset(folder, dataset):
    url = f"https://www.chrsmrrs.com/graphkerneldatasets/{dataset}.zip"
    zip_file = f"{dataset}.zip"
    files_path = os.path.join(folder, dataset)

    download_url(url, folder, zip_file)
    extract_zip(os.path.join(folder, zip_file), folder)

    files = glob.glob(os.path.join(files_path, f'{dataset}_*.txt'))
    names = [f.split(os.sep)[-1][len(dataset) + 1:-4] for f in files]

    adj = read_file(os.path.join(files_path, f"{dataset}_A.txt"))
    adj = [x.split(",") for x in adj]
    senders = jnp.array([int(x[0]) for x in adj]) - 1
    receivers = jnp.array([int(x[1]) for x in adj]) - 1

    batch = read_file(os.path.join(files_path, f"{dataset}_graph_indicator.txt"))
    batch = jnp.array([int(x) for x in batch]) - 1
    num_nodes = jnp.bincount(batch)
    aux = batch[senders]
    num_edges = jnp.bincount(aux)

    node_label = jnp.empty((batch.shape[0], 0))
    if 'node_labels' in names:
        node_label_l = read_file(os.path.join(files_path, f"{dataset}_node_labels.txt"))
        node_label_l = [[int(x) for x in label.split(",")] for label in node_label_l]
        node_label = jnp.array(node_label_l)
        if node_label.ndim == 1:
            node_label = node_label[:, None]
        node_label = node_label - jnp.min(node_label, axis=0)[0]
        # Note: num_classes is casted to int because one_hot requires static num_classes
        num_classes = int(jnp.max(node_label) + 1)  # Also count 0
        node_label = [jax.nn.one_hot(x, num_classes) for x in node_label]
        if len(node_label) == 1:  # turn list into array
            node_label = node_label[0]
        else:
            node_label = jnp.concatenate(node_label, axis=0)

    node_attr = jnp.empty((batch.shape[0], 0))
    if 'node_attributes' in names:
        node_attr_l = read_file(os.path.join(files_path, f"{dataset}_node_attributes.txt"))
        node_attr_l = [[float(x) for x in attr.split(",")] for attr in node_attr_l]
        node_attr = jnp.array(node_attr_l)
        if node_attr.ndim == 1:
            node_attr = node_attr[:, None]

    edge_label = jnp.empty((senders.shape[0], 0))
    if 'edge_labels' in names:
        edge_label_l = read_file(os.path.join(files_path, f"{dataset}_edge_labels.txt"))
        edge_label_l = [[int(x) for x in label.split(",")] for label in edge_label_l]
        edge_label = jnp.array(edge_label_l)
        if edge_label.ndim == 1:
            edge_label = edge_label[:, None]
        edge_label = edge_label - jnp.min(edge_label, axis=0)[0]
        num_classes = int(jnp.max(edge_label) + 1)  # Also count 0
        edge_labels = [jax.nn.one_hot(x, num_classes) for x in edge_label]
        if len(edge_labels) == 1:  # turn list into array
            edge_label = edge_labels[0]
        else:
            edge_label = jnp.concatenate(edge_labels, axis=0)

    edge_attr = jnp.empty((senders.shape[0], 0))
    if 'edge_attributes' in names:
        edge_attr_l = read_file(os.path.join(files_path, f"{dataset}_edge_attributes.txt"))
        edge_attr_l = [[float(x) for x in attr.split(",")] for attr in edge_attr_l]
        edge_attr = jnp.array(edge_attr_l)
        if edge_attr.ndim == 1:
            edge_attr = edge_attr[:, None]

    x = jnp.concatenate([node_attr, node_label], axis=-1)
    edge_attr = jnp.concatenate([edge_attr, edge_label], axis=-1)

    if x.shape[-1] == 0:
        x = None
    if edge_attr.shape[-1] == 0:
        edge_attr = None

    y = None
    if 'graph_labels' in names:
        y = read_file(os.path.join(files_path, f"{dataset}_graph_labels.txt"))
        y = jnp.array([int(x) for x in y])
    elif 'graph_attributes' in names:
        y = read_file(os.path.join(files_path, f"{dataset}_graph_attributes.txt"))
        y = jnp.array([float(x) for x in y])

    graph = DataGraphTuple(
        nodes=x,
        senders=senders,
        receivers=receivers,
        edges=edge_attr,
        n_node=num_nodes,
        n_edge=num_edges,
        globals=None,
        position=None,
        y=None,
        train_mask=None,
    )

    graphs = hkg.utils.unbatch(graph)
    return graphs, y