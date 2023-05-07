import pickle
import jax
import jax.numpy as jnp
from haiku_geometric.datasets.base import DataGraphTuple, GraphDataset
from haiku_geometric.datasets.utils import download_url, extract_zip


class Planetoid(GraphDataset):
    r"""The Planetoid dataset from the `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.

    Parameters:
        name (str): Name of the dataset. Can be one of ``'cora'``, ``'citeseer'`` or ``'pubmed'``.
        root (str): Root directory where the dataset will be saved.
        split (str): Which split to use. Can be one of ``'public'``, ``'full'`` or ``'random'``.
        num_train_per_class (int): Number of training examples for the ``'random'`` split.
        num_val (int): Number of validation examples. Only used for the ``'random'`` split.
        num_test (int): Number of test examples. Only used for the ``'random'`` split.

    **Attributes:**

        - **data**: (List[DataGraphTuple]): List of graph tuples (in this case only one graph).
        - **train_mask**: (List[bool]): Boolean mask for the training set.
        - **val_mask**: (List[bool]): Boolean mask for the validation set.
        - **test_mask**: (List[bool]): Boolean mask for the test set.
        - **num_classes**: (int): Number of classes.

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #nodes
              - #edges
              - #node features
              - #classes
            * - Cora
              - 2,708
              - 10,858
              - 1,433
              - 7
            * - CiteSeer
              - 3,312
              - 9,464
              - 3,703
              - 6
            * - PubMed
              - 19,717
              - 88,676
              - 500
              - 3
    
    """
    def _download_planetoid(self, dataset, folder):
        URL = "https://github.com/kimiyoung/planetoid/raw/master/data/"

        NAMES = ['x', 'y', 'tx', 'ty', 'graph', 'allx', 'ally', 'test.index']
        OBJECTS = []
        for i in range(len(NAMES)):
            download_url(f"{URL}ind.{dataset}.{NAMES[i]}", folder=folder, filename=None)
            if NAMES[i] == 'test.index':
                fb = open(folder + "ind.{}.{}".format(dataset, NAMES[i]), 'r')
                OBJECTS.append([int(x) for x in fb.readlines()])
            else:
                fb = open(folder + "ind.{}.{}".format(dataset, NAMES[i]), 'rb')
                OBJECTS.append(pickle.load(fb, encoding='latin1'))
        return tuple(OBJECTS)


    def _senders_receivers_from_dict(self, graph_dict):
        row, col = [], []
        for key, value in graph_dict.items():
            row += [key] * len(value)
            col += value
        #: TODO: remove self edges?
        return jnp.asarray(row), jnp.asarray(col)


    def _process_planetoid_data(self, x, y, tx, ty, graph, allx, ally, test_index):
        train_index = jnp.arange(y.shape[0], dtype=jnp.int32)
        val_index = jnp.arange(y.shape[0], y.shape[0] + 500, dtype=jnp.int32)
        test_index = jnp.array(test_index)
        sorted_test_index = jnp.sort(test_index)

        x = jnp.array(x.toarray())
        tx = jnp.array(tx.toarray())
        allx = jnp.array(allx.toarray())

        nx = jnp.concatenate([allx, tx], axis=0)
        ny = jnp.concatenate([ally, ty], axis=0).argmax(axis=1)

        nx = nx.at[test_index].set(nx[sorted_test_index])
        ny = ny.at[test_index].set(ny[sorted_test_index])

        def sample_mask(index, num_nodes):
            mask = jnp.zeros((num_nodes, ), dtype=jnp.uint8)
            mask = mask.at[index].set(1)
            mask = mask.astype(jnp.bool_)
            return mask

        train_mask = sample_mask(train_index, num_nodes=ny.shape[0])
        val_mask = sample_mask(val_index, num_nodes=ny.shape[0])
        test_mask = sample_mask(test_index, num_nodes=ny.shape[0])

        senders, receivers = self._senders_receivers_from_dict(graph)

        graph = DataGraphTuple(
            nodes=nx,
            senders=senders,
            receivers=receivers,
            edges=None,
            n_node=jnp.asarray([ny.shape[0]]),
            n_edge=jnp.asarray([senders.shape[0]]),
            globals=None,
            y=ny,
            train_mask=None,
            position=None
        )

        train_mask = train_mask
        val_mask = val_mask
        test_mask = test_mask
        num_classes = ally.shape[1]
        return graph, train_mask, val_mask, test_mask, num_classes
    
    def __init__(self, name: str, root: str, split: str = "public",
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000):
        x, y, tx, ty, graph, allx, ally, test_index = self._download_planetoid(name, root)
        graph, train_mask, val_mask, test_mask, num_classes \
                = self._process_planetoid_data(x, y, tx, ty, graph, allx, ally, test_index)
        super().__init__([graph])
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_val = num_val
        self.num_classes = num_classes
        
        split = split.lower()
        assert split in ['public', 'full', 'random']

        if split == 'full':
            self.train_mask = jnp.full(self.train_mask.shape, 1)
            self.train_mask.at[self.val_mask | self.test_mask].set(0)

        elif split == 'random':
            self.train_mask = jnp.full(self.train_mask.shape, 0)
            for c in range(self.num_classes):
                idx = jnp.nonzero(self.y == c)[0]
                idx = idx[
                    jax.random.permutation(
                        jax.random.PRNGKey(42), idx.shape[0])[:num_train_per_class]]
                self.train_mask.at[idx].set(1)

            remaining = jnp.nonzero(~self.train_mask)[0]
            remaining = remaining[jax.random.permutation(
                        jax.random.PRNGKey(42), remaining.shape[0])]

            self.val_mask = jnp.full(self.val_mask.shape, 0)
            self.val_mask.at[remaining[:num_val]].set(1)

            self.test_mask = jnp.full(self.test_mask.shape, 0)
            self.test_mask.at[remaining[:num_val]].set(1)