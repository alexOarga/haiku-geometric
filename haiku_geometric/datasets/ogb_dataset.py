from tqdm import tqdm
import jraph
import jax.numpy as jnp
from haiku_geometric.datasets.base import DataGraphTuple, GraphDataset
from typing import List, Tuple, Optional, Any


class OGB(GraphDataset):
    r"""Interface for the datasets from the `"Open Graph Benchmark"
    <https://ogb.stanford.edu/>`_ .


    .. note::
        Usage of this dataset requires installing first the `ogb <https://github.com/snap-stanford/ogb>`_ package:
            .. code-block:: bash

                pip install ogb

    Parameters:
        - name (str): Name of the OGB dataset.
        - root (str): Root directory where the dataset should be saved.


    **Attributes:**

    - **data**: (List[DataGraphTuple]): List of graph tuples.
    - **splits**: (Dict[str, List[int]]): Dictionary with the indices of the train, validation and test set:

        .. code-block:: python

            {
                'train': [...],
                'valid': [...],
                'test': [...]
            }
    """

    # function from: https://github.com/tensorflow/gnn/blob/cf931728df08fb379e624f30fa13bca73b32c4c7/tensorflow_gnn/converters/ogb/convert_ogb_dataset.py
    def _create_dataset(self, dataset: str, datasets_root: Optional[str] = None) -> Any:

        # Optional imports
        import ogb
        from ogb.graphproppred import GraphPropPredDataset
        from ogb.nodeproppred import NodePropPredDataset
        from ogb.linkproppred import LinkPropPredDataset

        problem_type = dataset.split("-")[0]
        kwargs = dict(name=dataset, root=datasets_root)
        if problem_type == "ogbn":
            dataset = ogb.nodeproppred.NodePropPredDataset(**kwargs)
        elif problem_type == "ogbl":
            dataset = ogb.linkproppred.LinkPropPredDataset(**kwargs)
        elif problem_type == "ogbg":
            dataset = ogb.graphproppred.GraphPropPredDataset(**kwargs)
        else:
            raise ValueError("Invalid problem type for {}".format(dataset))
        return dataset
    
    
    def _convert_ogb_graph(self, graph):
        y = graph[1]
        graph = graph[0]

        args = {
            'nodes': jnp.asarray(graph['node_feat']),
            'edges': jnp.asarray(graph['edge_feat']),
            'senders': jnp.asarray(graph['edge_index'][0]),
            'receivers': jnp.asarray(graph['edge_index'][1]),
            'n_node': jnp.asarray([graph['num_nodes']]),
            'n_edge': jnp.asarray([graph['edge_index'].shape[1]]),
            'globals': None,
            'y': jnp.asarray(y),
            'train_mask': None,
            'position': None
        }
        return DataGraphTuple(**args)
    
    
    def get_idx_split(self):
        return self.splits
    
    def __init__(self, name: str, root: Optional[str] = None):
        """"""
        dataset = self._create_dataset(name, root)
        
        splits = dataset.get_idx_split()
        self.splits = {
            'train': jnp.asarray(splits['train']),
            'valid': jnp.asarray(splits['valid']),
            'test': jnp.asarray(splits['test'])
        }
        
        res = []
        bar = tqdm(range(len(dataset)))
        for i in bar:
            bar.set_description("Processing graph ")
            res.append(self._convert_ogb_graph(dataset[i]))

        super().__init__(res)