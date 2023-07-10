
from tqdm import tqdm
import pickle
import os, sys
import os.path as osp
import jax.numpy as jnp
from haiku_geometric.datasets.base import DataGraphTuple, GraphDataset
from haiku_geometric.datasets.utils import pickle_save_object, download_url, extract_zip


class GNNBenchmarkDataset(GraphDataset):
    r"""Interface for the datasets from the `"Benchmarking Graph Neural Networks"
    <arxiv.org/abs/2003.00982>`_ .

    .. note::
        Usage of this dataset requires installing first the
        `Pytorch Geometric <https://github.com/pyg-team/pytorch_geometric>`_ package.


    Parameters:
        name (str): Name of the GNN Benchmark dataset. Available datasets are: ``PATTERN``, ``CLUSTER``, ``MNIST``, ``CIFAR10``, ``TSP`` and ``CSL``.
        root (str): Root directory where the dataset should be saved.
        split (str): Split of the dataset. Split can take values: ``train``, ``valid`` and ``test``.


    **Attributes:**

        - **data** (List[DataGraphTuple]): List of graph tuples.
    """
    _root_url = 'https://data.pyg.org/datasets/benchmarking-gnns'
    _urls = {
        'PATTERN': f'{_root_url}/PATTERN_v2.zip',
        'CLUSTER': f'{_root_url}/CLUSTER_v2.zip',
        'MNIST': f'{_root_url}/MNIST_v2.zip',
        'CIFAR10': f'{_root_url}/CIFAR10_v2.zip',
        'TSP': f'{_root_url}/TSP_v2.zip',
        'CSL': 'https://www.dropbox.com/s/rnbkp5ubgk82ocu/CSL.zip?dl=1',
    }
    
    def _process_gnn_benchmark_data(self, data):
        def _transform_graph(graph):
            return DataGraphTuple(
                nodes=jnp.array(graph['x'].numpy()),
                position=jnp.array(graph['pos'].numpy()),
                y=jnp.array(graph['y']),
                edges=jnp.array(graph['edge_attr'].numpy()),
                senders=jnp.array(graph['edge_index'].numpy()[0]),
                receivers=jnp.array(graph['edge_index'].numpy()[1]),
                n_node=jnp.array([graph['x'].shape[0]]),
                n_edge=jnp.array([graph['edge_index'].shape[1]]),
                globals=None,
                train_mask=None,
            )

        bar = tqdm(range(len(data)))
        for i in bar:
            bar.set_description("Processing graph ")
            data[i] = _transform_graph(data[i])
        return data
    
    
    def _raw_names(self, name):
            if name == 'CSL':
                return [
                    'graphs_Kary_Deterministic_Graphs.pkl',
                    'y_Kary_Deterministic_Graphs.pt'
                ]
            else:
                name = self._urls[name].split('/')[-1][:-4]
                return [f'{name}.pt']

    
    def _processed_path(self, root, name):
        raw_file_names = self._raw_names(name)
        path = osp.join(root, raw_file_names[0])
        self.processed_files = {
            'train': f"{path}.train.obj",
            'val': f"{path}.val.obj",
            'test': f"{path}.test.obj",
        }
        return self.processed_files
            
            
    def _process_and_save(self, root, name):

        # Optional imports
        import torch

        raw_file_names = self._raw_names(name)
        path = osp.join(root, raw_file_names[0])
        ys = torch.load(path)
        os.unlink(path)
        train = self._process_gnn_benchmark_data(ys[0])
        val = self._process_gnn_benchmark_data(ys[1])
        test = self._process_gnn_benchmark_data(ys[2])
        
        print(f'Saving split files', file=sys.stderr)
        pickle_save_object(train, self.processed_files['train'])
        pickle_save_object(val, self.processed_files['val'])
        pickle_save_object(test, self.processed_files['test'])
            
    
    def _process_cls(self, root, name):

        # Optional imports
        import torch

        raw_paths = self._raw_names(name)
        path = download_url(self._urls[name], folder=root, filename=None)
        extract_zip(path, root)
        path = osp.join(root, raw_paths[0])
        with open(path, 'rb') as f:
            adjs = pickle.load(f)
        path = osp.join(root, raw_paths[1])  
        ys = torch.load(path).tolist()

        #: TODO: remove self edges
        res = []
        for adj, y in zip(adjs, ys):
            res.append(
                DataGraphTuple(
                    nodes=None,
                    edges=None,
                    y=jnp.array(y),
                    senders=jnp.array(adj.row),
                    receivers=jnp.array(adj.col),
                    n_node=jnp.array([adj.shape[0]]),
                    n_edge=jnp.array([adj.row.shape[0]]),
                    globals=None,
                    position=None,
                    train_mask=None,
                )
            )

        return res
    
            
    def __init__(self, name: str, root: str, split: str = "train"):

        names = ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10', 'TSP', 'CSL']
        if name not in names:
            raise ValueError(f"Database name '{name}' not available. Avilable databases are: "
                            f"{names}")
        
        if name == 'CSL':
            dataset = self._process_cls(root, name)
            super().__init__(dataset)
            
        else:
            if split not in ['train', 'val', 'test']:
                raise ValueError(f"Split '{split}' not supported. Avilable splits are: "
                                 f"'train', 'val', or 'test'")

            self._processed_path(root, name)
            if osp.exists(self.processed_files[split]):
                print(f"Using existing split files '{self.processed_files}'", file=sys.stderr)
                dataset = pickle.load(open(self.processed_files[split],'rb'))
                super().__init__(dataset)

            else:
                print(f"Downloading and processing new dataset", file=sys.stderr)
                path = download_url(self._urls[name], folder=root, filename=None)
                extract_zip(path, root)
                self._process_and_save(root, name)
                dataset = pickle.load(open(self.processed_files[split],'rb'))
                super().__init__(dataset)
            