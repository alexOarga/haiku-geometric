import haiku as hk
import jraph
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import warnings

from typing import Optional, Union, List, Callable, Dict, Any
from .pna_conv import PNAConv
from .gat_conv import GATConv
from .gine_conv import GINEConv
from haiku_geometric.nn import SelfAttention

class GPSLayer(hk.Module):
    """GPS layer from the `"Recipe for a General, Powerful, Scalable Graph Transformer"
    <https://arxiv.org/pdf/2205.12454.pdf>`_ paper.
    
    🚧: This layer is still under development and might not work as expected.

    Args:
        dim_h (int): Size of each output features.
        local_gnn_type (str): Name of a message passing neural network.
            Available networks are: :obj:`None`, :obj:`"GINE"`, :obj:`"GAT"`,
            :obj:`"PNA"`.
        global_model_type (str): Name of a global attention layer.
            Available networks are: :obj:`None`, :obj:`"Transformer"`, :obj:`"Performer"`.
        act: (Callable): activation function (e.g. :obj:`jax.nn.relu`).
        num_heads (int, optional): number of heads when using multi-head attention.
            (default: :obj:`1`).
        pna_degrees (jnp.ndarray, optional): Array of degrees histogram when using PNA.
        equivstable_pe (bool, optional): * Not implemented *.
            (default: :obj:`False`).
        dropout (float, optional): dropout rate.
            (default: :obj:`0.0`).
        attn_dropout (float, optional): dropout rate with global attention.
            (default: :obj:`0.0`).
        layer_norm (bool, optional): Whether to use layer normalization.
            (default: :obj:`False`).
        batch_norm (bool, optional): Whether to use batch normalization.
            (default: :obj:`True`).
    """

    def __init__(self, dim_h: int, local_gnn_type: str, global_model_type: str, 
                 act: Callable, num_heads: int=1, pna_degrees: jnp.ndarray=None, 
                 equivstable_pe:bool=False, dropout: float=0.0, attn_dropout: float=0.0, 
                 layer_norm: bool=False, batch_norm: bool=True):
        """"""
        super().__init__()
        
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = act
        self.local_gnn_type = local_gnn_type
        self.global_model_type = global_model_type
        self.dropout = dropout
        self.attn_dropout = attn_dropout
            
        # Initialize message passing layer
        if local_gnn_type is None:
            self.local_model = None
        elif local_gnn_type == 'GENConv':
            raise NotImplementedError("GENConv is not yet supported for GPSLayer")
        elif local_gnn_type == 'GINE':
            gin_nn = hk.Sequential(hk.Linear(dim_h),
                                   self.activation,
                                   hk.Linear(dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                raise NotImplementedError("Specialised GINE equivstable_pe is "
                                          "not yet supported for GPSLayer")
            else:
                self.local_model = GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            #: TODO: add edge_dim on GATConv
            print(dim_h, num_heads)
            self.local_model = GATConv(out_channels=(dim_h // num_heads),
                                        heads=num_heads)
        elif local_gnn_type == 'PNA':
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            self.local_model = PNAConv(out_channels=dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=pna_degrees,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            raise NotImplementedError("CustomGatedGCN not yet supported for GPSLayer")
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        
            
        # Initialize global attention model
        if global_model_type is None:
            self.self_attn = None
        elif global_model_type == 'Transformer':
            warnings.warn("Dropout is not currently supported with transformers")
            self.self_attn = hk.MultiHeadAttention(
                    num_heads=num_heads, 
                    key_size=dim_h, 
                    value_size=dim_h, 
                    model_size=dim_h,
                    w_init=hk.initializers.TruncatedNormal())
        elif global_model_type == 'Performer':
            raise NotImplementedError("Performer not yet supported for GPSLayer")
            # TODO: Include google's JAX performer implementation
            '''
            self.self_attn = SelfAttention(
                    num_heads=num_heads, 
                    key_size=dim_h, 
                    value_size=dim_h, 
                    model_size=dim_h,
                    w_init=hk.initializers.TruncatedNormal())
            '''
        elif global_model_type == "BigBird":
            raise NotImplementedError("BigBird not yet supported for GPSLayer")
        else:
            raise ValueError(f"Unsupported global attention model: "
                             f"{global_model_type}")
        
        # Initialize normalization layers
        if self.layer_norm:
            self.norm1_local = hk.LayerNorm(axis=-1, param_axis=-1,
                    create_scale=True, create_offset=True)
            self.norm1_attn = hk.LayerNorm(axis=-1, param_axis=-1,
                    create_scale=True, create_offset=True)
            
        if self.batch_norm:
            self.norm1_local = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)
            self.norm1_attn = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)
        #self.dropout_local = nn.Dropout(dropout)
        #self.dropout_attn = nn.Dropout(dropout)
        
        # Initialize Feed Forward block
        self.ff_linear1 = hk.Linear(dim_h * 2)
        self.ff_linear2 = hk.Linear(dim_h)
        if self.layer_norm:
            self.norm2 = hk.LayerNorm(axis=-1, param_axis=-1,
                    create_scale=True, create_offset=True)
        if self.batch_norm:
            self.norm2 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)
        #self.ff_dropout1 = nn.Dropout(dropout)
        #self.ff_dropout2 = nn.Dropout(dropout)
    
    def __call__(self,
                 training: bool,
                 nodes: jnp.ndarray,
                 senders: jnp.ndarray = None,
                 receivers: jnp.ndarray = None,
                 edges: Optional[jnp.ndarray] = None,
                 num_nodes: Optional[int] = None
                 ) -> jnp.ndarray:
        """"""
        h = nodes
        h_in1 = h

        h_out_list = []

        if num_nodes is None:
            num_nodes = tree.tree_leaves(h)[0].shape[0]

        # Local MPNN
        if self.local_model is not None:
            h_local = self.local_model(h, senders, receivers, edges, num_nodes=num_nodes)
            if training:
                h_local = hk.dropout(jax.random.PRNGKey(42), self.dropout, h_local)
            else:
                #: TODO: might be necessary to scale weights
                pass
            h_local = h_in1 + h_local  # Residual connection.
            
            if self.layer_norm:
                h_local = self.norm1_local(h_local)
            if self.batch_norm:
                h_local = self.norm1_local(h_local, training)
            h_out_list.append(h_local)
            
        # Multi-head attention.
        if self.self_attn is not None:
            #: TODO: implement to_dense.batch (?)
            # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/to_dense_batch.html#to_dense_batch
            if self.global_model_type == 'Transformer':
                #: TODO: define mask
                h_attn = self.self_attn(h, h, h)
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h)
            elif self.global_model_type == 'BigBird':
                raise NotImplementedError("BigBird not yet supported for GPSLayer")
            else:
                raise ValueError(f"Unsupported global attention model: "
                             f"{self.global_model_type}")
                
            if training:
                h_attn = hk.dropout(jax.random.PRNGKey(42), self.attn_dropout, h_attn)
            else:
                #: TODO: might be necesary to scale weights
                pass
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn, training)
            h_out_list.append(h_attn)
            
            # Combine local and global outputs.
            h = sum(h_out_list)

        # Feed Forward block
        h = h + self._ff_block(h, training)
        if self.layer_norm:
            h = self.norm2(h)
        if self.batch_norm:
            h = self.norm2(h, training)

        return h
    
    def _ff_block(self, x, training):
        
        x = self.activation(self.ff_linear1(x))
        if training:
            x = hk.dropout(jax.random.PRNGKey(42), self.dropout, x)
        else:
            #: TODO: might be necesary to scale weights
            pass
        
        x = self.ff_linear2(x)
        if training:
            x = hk.dropout(jax.random.PRNGKey(42), self.dropout, x)
        else:
            #: TODO: might be necesary to scale weights
            pass
        
        return x