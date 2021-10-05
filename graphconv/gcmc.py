import torch
from torch import nn

import dgl
from dgl import function as fn

from utils import activation_map

class GCMCConv(nn.Module):
    def __init__(self,
                in_feats_dim,
                out_feats_dim,
                drop_out = 0.):
        """GCMC Convolution

        Paramters
        ---------
        in_feats_dim : int
            dimension of input features
        out_feats_dim : int
            dimension of output features
        drop_out : float
            dropout rate (neighborhood dropout)
        """
        super().__init__()

        self.feature_transform = nn.Linear(in_feats_dim, out_feats_dim)
        self.dropout = nn.Dropout(drop_out)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.feature_transform.weight)

    def forward(self, graph, feats):
        """Apply GCMC Convoluiton to specific edge type {r}

        Paramters
        ---------
        graph : dgl.graph
        src_feats : torch.FloatTensor
            source node features

        ci : torch.LongTensor
            in-degree of sources ** (-1/2)
            shape : (n_sources, 1)
        cj : torch.LongTensor
            out-degree of destinations ** (-1/2)
            shape : (n_destinations, 1)

        Returns
        -------
        output : torch.FloatTensor
            output features

        Notes
        -----
        1. message passing
            MP_{j -> i, r} = W * feature_{j} / ( N_{i, r} * N_{j, r} )
                where N_{i} ; number of neighbors_{i, r} ** (1/2)
        2. aggregation
            \sum_{j \in N(i), r} MP_{j -> i, r}
        """
        if isinstance(feats, tuple):
            src_feats, dst_feats = feats

        with graph.local_scope():
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']

            src_feats = self.feature_transform(src_feats)

            cj_dropout = self.dropout(cj)
            weighted_feats = torch.mul(src_feats, cj_dropout)
            graph.srcdata['h'] = weighted_feats

            graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'out'))
            out = torch.mul(graph.dstdata['out'], ci)
            
        return out

class GCMCLayer(nn.Module):
    def __init__(self,
                edge_types,
                user_feats_dim,
                item_feats_dim,
                out_feats_dim,
                agg = 'sum',
                drop_out = 0.,
                activation = 'relu'):
        super().__init__()
        """GCMC Layer

        edge_types : list
            all edge types
        user_feats_dim : int
            dimension of user features
        item_feats_dim : int
            dimension of item features
        out_feats_dim : int
            dimension of hidden features (output dimension)
        agg : str
            aggreration type
        activation : str
            activation function
        """
        assert agg in ['sum', 'stack', 'max', 'min', 'mean'], "Unsupported aggregation type."
        self.agg = agg

        if agg == 'sum':
            self.message_dim = out_feats_dim
        else:
            n_etypes = len(edge_types)
            assert out_feats_dim % n_etypes == 0, f"out_feats_dim must be a multiple of {n_etypes} ( len(edge_types) )"
            self.message_dim = out_feats_dim // n_etypes
        self.out_feats_dim = out_feats_dim

        conv = {}
        for edge in edge_types:
            user_to_item_key = f'{edge}'
            item_to_user_key = f'reverse-{edge}'

            # convolution on user -> item graph
            conv[user_to_item_key] = GCMCConv(in_feats_dim = user_feats_dim,
                                            out_feats_dim = self.message_dim,
                                            drop_out = drop_out)
            
            # convolution on item -> user graph
            conv[item_to_user_key] = GCMCConv(in_feats_dim = item_feats_dim,
                                            out_feats_dim = self.message_dim,
                                            drop_out = drop_out)

        self.conv = dgl.nn.pytorch.HeteroGraphConv(conv, aggregate = agg)
        self.activation_agg = activation_map[activation]
        self.feature_dropout = nn.Dropout(drop_out)

    def flatten(self, feats):
        """
        if agg == 'stack':
            conv out : (n_users, n_edges, message_feats_dim)
            returns : (n_users, n_edges * message_feats_dim) i.e. (n_users, out_feats_dim)
        else:
            conv out : (n_users, out_feats_dim)
            returns : (n_users, out_feats_dim)
        """
        return feats.contiguous().view(-1, self.out_feats_dim)

    def forward(self, graph, ufeats, ifeats, ukey = 'user', ikey = 'item'):
        """
        Paramters
        ---------
        graph : dgl.graph
        ufeats, ifeats : torch.FloatTensor
            node features
        ukey, ikey : str
            target node types

        Returns
        -------
        ufeats, ifeats : torch.FloatTensor
            output features

        Notes
        -----
        1. message passing
            MP_{i} = \{ MP_{i, r_{1}}, MP_{i, r_{2}}, ... \}
        2. aggregation
            h_{i} = \sigma( aggregate( MP_{i} ) )
        """
        feats = {
            ukey : ufeats,
            ikey : ifeats
            }

        out = self.conv(graph, feats)

        ufeats = self.flatten(out[ukey])
        ufeats = self.activation_agg(ufeats)
        ufeats = self.feature_dropout(ufeats)

        ifeats = self.flatten(out[ikey])
        ifeats = self.activation_agg(ifeats)
        ifeats = self.feature_dropout(ifeats)

        return ufeats, ifeats