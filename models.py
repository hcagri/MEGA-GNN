import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv, to_hetero
from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import HeteroData
import torch.nn.functional as F
import torch
import logging
import numpy as np
from torch_scatter import scatter
from torch_geometric.utils import degree
from genagg import GenAgg
from genagg.MLPAutoencoder import MLPAutoencoder
import math 
import time


class DiscreteEncoder(nn.Module):
    def __init__(self, hidden_channels, max_num_features=10, max_num_values=500): #10
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(max_num_values, hidden_channels) 
                    for i in range(max_num_features)])

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()
            
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = 0
        for i in range(x.size(1)):
            out = out + self.embeddings[i](x[:, i])
        return out
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerAgg(nn.Module):
    def __init__(self, d_model = 66):
        super().__init__()

        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=0.05, max_len=128)
        self.trans_enc = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=2, 
            dim_feedforward=d_model*4, 
            dropout=0.0,
            batch_first=True, # If True, then the input and output tensors are provided as (batch, seq, feature). 
            norm_first=False
            )

    def forward(self, x, index, timestamps_):
        # Add timestamps to the edge features to be able to sort according to them
        x = torch.cat([timestamps_.view(-1, 1), x], dim=1)
        
        s_pre = time.time()
        sort_ids = torch.argsort(index)
        dense_edge_feats, mask = to_dense_batch(x[sort_ids, :], index[sort_ids])
        sorted_dense_edge_feats, sorted_mask = self.sort_wrt_time(dense_edge_feats, mask)
        
        s_post = time.time()
        sorted_dense_edge_feats = self.pos_enc(sorted_dense_edge_feats.permute(1,0,2)).permute(1,0,2)
        sorted_dense_edge_feats = self.trans_enc(sorted_dense_edge_feats, src_key_padding_mask = ~sorted_mask)
        sorted_dense_edge_feats[~sorted_mask.unsqueeze(-1).expand(-1, -1, sorted_dense_edge_feats.shape[-1])] = 0
        s_res = time.time()

        logging.debug(f"Preprocessing Time Transformer: {s_post - s_pre} | Transformer forward pass: {s_res - s_post}")

        return sorted_dense_edge_feats.mean(dim=1).squeeze()

    def sort_wrt_time(self, matt, mask):
        first_feature = matt[:, :, 0] 
        sort_indices = torch.argsort(first_feature, dim=1)
        sorted_matt = torch.gather(matt, 1, sort_indices.unsqueeze(-1).expand(-1, -1, matt.shape[-1]))
        sorted_mask = torch.gather(mask, 1, sort_indices)
        return sorted_matt[:, :, 1:], sorted_mask

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class GRUAgg(nn.Module):
    def __init__(self, d_model = 66):
        super().__init__()

        self.gru = nn.GRU(
            d_model, 
            hidden_size=d_model, 
            num_layers=2, 
            batch_first=True
            )

    def forward(self, x, index, timestamps_):
        # Add timestamps to the edge features to be able to sort according to them
        x = torch.cat([timestamps_.view(-1, 1), x], dim=1)
        
        sort_ids = torch.argsort(index)
        dense_edge_feats, mask = to_dense_batch(x[sort_ids, :], index[sort_ids])
        sorted_dense_edge_feats, sorted_mask = self.sort_wrt_time(dense_edge_feats, mask)

        sorted_dense_edge_feats = self.gru(sorted_dense_edge_feats)[0]
        # sorted_dense_edge_feats[~sorted_mask.unsqueeze(-1).expand(-1, -1, sorted_dense_edge_feats.shape[-1])] = 0

        return sorted_dense_edge_feats.mean(dim=1).squeeze()

    def sort_wrt_time(self, matt, mask):
        first_feature = matt[:, :, 0] 
        sort_indices = torch.argsort(first_feature, dim=1)
        sorted_matt = torch.gather(matt, 1, sort_indices.unsqueeze(-1).expand(-1, -1, matt.shape[-1]))
        sorted_mask = torch.gather(mask, 1, sort_indices)
        return sorted_matt[:, :, 1:], sorted_mask

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
class PnaAgg(nn.Module):
    def __init__(self , n_hidden, deg):
        super().__init__()
        
        aggregators = ['mean', 'min', 'max', 'std']
        self.num_aggregators = len(aggregators)
        scalers = ['identity', 'amplification', 'attenuation']

        self.agg = DegreeScalerAggregation(aggregators, scalers, deg)
        self.lin = nn.Linear(len(scalers)*len(aggregators)*n_hidden, n_hidden)

    def forward(self, x, index):
        out = self.agg(x, index)
        return self.lin(out)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                param = nn.init.kaiming_normal_(param.detach())
            elif 'bias' in name:
                param = nn.init.constant_(param.detach(), 0)
        
class IdentityAgg(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, index):
        return x

class GinAgg(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.nn = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(), 
                nn.Linear(n_hidden, n_hidden)
                )
    def forward(self, x, index):
        out = scatter(x, index, dim=0, reduce='sum')
        return self.nn(out)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass

class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=True, bias=True, activation_fn="relu", LN='False'):
        super().__init__()
        n_hid = nout
        self.layers = nn.ModuleList([nn.Linear(nin if i==0 else n_hid, 
                                     n_hid if i<nlayer-1 else nout, 
                                     bias=True if (i==nlayer-1 and not with_final_activation and bias) # TODO: revise later
                                        or (not with_norm) else False) # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i<nlayer-1 else nout) if not LN else nn.LayerNorm(n_hid) if with_norm else Identity()
                                     for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin==nout) ## TODO: test whether need this
        self.act = getattr(F, activation_fn)

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = self.act(x)  

        return x
    
class AdammAgg(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.edge_transform = MLP(n_hidden, n_hidden, nlayer=1, with_final_activation=False)
               
    def forward(self, x, index):
        out = scatter(x, index, dim=0, reduce='sum')
        return self.edge_transform(out)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class SumAgg(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, index):
        return scatter(x, index, dim=0, reduce='sum')

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class MultiEdgeAggModule(nn.Module):
    def __init__(self, n_hidden=None, agg_type=None, index=None):
        super().__init__()
        self.agg_type = agg_type

        if agg_type == 'genagg':
            self.agg = GenAgg(f=MLPAutoencoder, jit=False)
        elif agg_type == 'gin':
            self.agg = GinAgg(n_hidden=n_hidden)
        elif agg_type == 'pna':
            uniq_index, inverse_indices = torch.unique(index, return_inverse=True)
            d = degree(inverse_indices, num_nodes=uniq_index.numel(), dtype=torch.long)
            deg = torch.bincount(d, minlength=1)
            self.agg = PnaAgg(n_hidden=n_hidden, deg=deg)
        elif agg_type == 'sum':
            self.agg = SumAgg()
        elif agg_type == 'transformer':
            self.agg = TransformerAgg(d_model=n_hidden)
        elif agg_type == 'gru':
            self.agg = GRUAgg(d_model=n_hidden)
        elif agg_type == 'adamm':
            self.agg = AdammAgg(n_hidden)
        else:
            self.agg = IdentityAgg()
        
    def forward(self, edge_index, edge_attr, simp_edge_batch, times):
        _, inverse_indices = torch.unique(simp_edge_batch, return_inverse=True)
        new_edge_index = scatter(edge_index, inverse_indices, dim=1, reduce='mean') if self.agg_type is not None else edge_index

        if (self.agg_type == 'transformer') or (self.agg_type == 'gru'):
            new_edge_attr = self.agg(x= edge_attr, index=inverse_indices, timestamps_=times)
        else:
            new_edge_attr = self.agg(x=edge_attr, index=inverse_indices)
        return new_edge_index, new_edge_attr, inverse_indices
    
    def reset_parameters(self):
        self.agg.reset_parameters()

class MultiMPNN(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, n_hidden=100, 
                 edge_updates=False,edge_dim=None, final_dropout=0.5, 
                 index_ = None, deg=None, args=None):
        super().__init__()
        self.args = args
        self.n_hidden = n_hidden
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        if args.reverse_mp or args.reverse_mp_lp:
            self.edge_emb_rev = nn.Linear(edge_dim, n_hidden)

        if args.edge_agg_type =='adamm':
            self.edge_direction_encoder = DiscreteEncoder(n_hidden, max_num_values=4)
    
        self.gnn = GnnHelper(num_gnn_layers=num_gnn_layers, n_hidden=n_hidden, edge_updates=edge_updates, final_dropout=final_dropout,
                             index_=index_, deg=deg, args=args)
        
        if args.reverse_mp or args.reverse_mp_lp:
            self.gnn  = to_hetero(self.gnn, metadata= (['node'], [('node', 'to', 'node'), ('node', 'rev_to', 'node')]), aggr='mean')

        if args.task == 'edge_class':
            self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                                Linear(25, n_classes))
        elif args.task == 'node_class':
            self.mlp = nn.Sequential(Linear(n_hidden, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                                Linear(25, n_classes))
        elif args.task == 'lp':
            if args.ports_batch:
                # positive and negative edges do not contain port information, since ports are assigned after neighborhood sampling.
                self.edge_emb_new = nn.Linear(edge_dim-2, n_hidden)
            self.edge_readout = LinkPredHead(n_hidden=n_hidden, n_classes=1, final_dropout=final_dropout)
    
    def forward(self, data):
        if isinstance(data, HeteroData):

            # Initial Embedding Layers
            x_dict = {"node": self.node_emb(data['node'].x)}
            edge_attr_dict = {
                    ("node", 'to', 'node'): self.edge_emb(data['node', 'to', 'node'].edge_attr), 
                    ("node", 'rev_to', 'node'): self.edge_emb_rev(data['node', 'rev_to', 'node'].edge_attr),    
                }
            simp_edge_batch_dict = data.simp_edge_batch_dict if self.args.flatten_edges else None

            # Message Passing Layers
            x_dict, edge_attr_dict = self.gnn(x_dict, data.edge_index_dict, edge_attr_dict, simp_edge_batch_dict)
            x = x_dict['node']
            edge_attr = edge_attr_dict['node', 'to', 'node']

            # Prediction Heads
            if self.args.task == 'edge_class':
                x = x[data['node', 'to', 'node'].edge_index.T].reshape(-1, 2*self.n_hidden).relu()
                x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
                out = self.mlp(x)
            elif self.args.task == 'node_class':
                out = self.mlp(x)
            elif self.args.task == 'lp':
                if self.args.ports_batch:
                    neg_edge_attr = self.edge_emb_new(data['node', 'to', 'node'].neg_edge_attr) 
                    pos_edge_attr = self.edge_emb_new(data['node', 'to', 'node'].pos_edge_attr)
                else:
                    neg_edge_attr = self.edge_emb(data['node', 'to', 'node'].neg_edge_attr) 
                    pos_edge_attr = self.edge_emb(data['node', 'to', 'node'].pos_edge_attr)
                out = self.edge_readout(x, data['node', 'to', 'node'].pos_edge_index, pos_edge_attr, data['node', 'to', 'node'].neg_edge_index, neg_edge_attr)

        else:
            # Initial Embedding Layers
            x = self.node_emb(data.x)
            edge_attr = self.edge_emb(data.edge_attr) 
            simp_edge_batch = data.simp_edge_batch if self.args.flatten_edges else None

            if self.args.edge_agg_type =='adamm':
                edge_attr = edge_attr + self.edge_direction_encoder(data.edge_direction)

            # Message Passing Layers
            x, edge_attr = self.gnn(x, data.edge_index, edge_attr, simp_edge_batch)

            # Prediction Heads
            if self.args.task == 'edge_class':
                x = x[data.edge_index.T].reshape(-1, 2*self.n_hidden).relu()
                x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
                out = self.mlp(x)
            elif self.args.task == 'node_class':
                out = self.mlp(x)
            elif self.args.task == 'lp':
                if self.args.ports_batch:
                    neg_edge_attr = self.edge_emb_new(data.neg_edge_attr) 
                    pos_edge_attr = self.edge_emb_new(data.pos_edge_attr)
                else:
                    neg_edge_attr = self.edge_emb(data.neg_edge_attr) 
                    pos_edge_attr = self.edge_emb(data.pos_edge_attr)
                out = self.edge_readout(x, data.pos_edge_index, pos_edge_attr, data.neg_edge_index, neg_edge_attr)
        return out

class LinkPredHead(torch.nn.Module):
    """Readout head for link prediction.

    Parameters
    ----------
    config : GNNConfig
        Architecture configuration
    """
    def __init__(self, n_hidden, n_classes, final_dropout) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.final_dropout = final_dropout

        self.mlp = nn.Sequential(Linear(self.n_hidden*3, self.n_hidden), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(self.n_hidden, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                            Linear(25, self.n_classes))
            
    def forward(self, x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr):
        #reshape s.t. each row in x corresponds to the concatenated src and dst node features for each edge
        x_pos = x[pos_edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x_neg = x[neg_edge_index.T].reshape(-1, 2 * self.n_hidden).relu()

        #concatenate the node feature vector with the corresponding edge features
        x_pos = torch.cat((x_pos, pos_edge_attr.view(-1, pos_edge_attr.shape[1])), 1)
        x_neg = torch.cat((x_neg, neg_edge_attr.view(-1, neg_edge_attr.shape[1])), 1)

        return (torch.sigmoid(self.mlp(x_pos)), torch.sigmoid(self.mlp(x_neg)))

class GnnHelper(torch.nn.Module):
    def __init__(self, num_gnn_layers, n_hidden=100, edge_updates=False, 
                final_dropout=0.5, index_ = None, deg = None, args=None):
        super().__init__()

        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout
        self.flatten_edges = args.flatten_edges
        self.edge_agg_type = args.edge_agg_type
        self.args = args

    
        if args.node_agg_type == 'genagg':
            self.node_agg = GenAgg(f=MLPAutoencoder, jit=False)
        elif args.node_agg_type == 'sum':
            self.node_agg = 'sum'
     
        self.edge_aggrs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            if args.model == 'gin':
                conv = GINEConv(nn.Sequential(
                    nn.Linear(self.n_hidden, self.n_hidden), 
                    nn.ReLU(), 
                    nn.Linear(self.n_hidden, self.n_hidden)
                    ), 
                    edge_dim=self.n_hidden, 
                    aggr = self.node_agg # Added New!!!
                    )
            elif args.model == 'pna':
                aggregators = ['mean', 'min', 'max', 'std']
                scalers = ['identity', 'amplification', 'attenuation']
                conv = PNAConv(in_channels=n_hidden, out_channels=n_hidden,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=n_hidden, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
                
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            if self.flatten_edges:
                edge_agg = MultiEdgeAggModule(n_hidden, agg_type=args.edge_agg_type, index=index_)
                self.edge_aggrs.append(edge_agg)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))
        

        if self.edge_agg_type == 'adamm':
            self.edge_aggrs = None
            self.edge_agg = MultiEdgeAggModule(n_hidden, agg_type=args.edge_agg_type, index=index_)
        

    def forward(self, x, edge_index, edge_attr, simp_edge_batch=None):
        times= None # TODO: Not implemented yet

        if self.edge_agg_type == 'adamm':
            # Apply the flattening at the beggining only.
            edge_index, edge_attr, _ = self.edge_agg(edge_index, edge_attr, simp_edge_batch, times)
            src, dst = edge_index

            for i in range(self.num_gnn_layers):
                x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
                if self.edge_updates: 
                    edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2   
        else:
            src, dst = edge_index

            for i in range(self.num_gnn_layers):
                if self.flatten_edges:
                    n_edge_index, n_edge_attr, inverse_indices  = self.edge_aggrs[i](edge_index, edge_attr, simp_edge_batch, times)
                    x = (x + F.relu(self.batch_norms[i](self.convs[i](x, n_edge_index, n_edge_attr)))) / 2
                    if self.edge_updates: 
                        remapped_edge_attr = torch.index_select(n_edge_attr, 0, inverse_indices) # artificall node attributes 
                        edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], remapped_edge_attr, edge_attr], dim=-1)) / 2
                else:
                    x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
                    if self.edge_updates: 
                        edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
                    
        return x, edge_attr

