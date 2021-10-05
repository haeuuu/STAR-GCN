import torch
import torch.nn as nn

import dgl
import dgl.function as fn

from graphconv.gcmc import GCMCLayer

from utils import activation_map

class GCMCEncoder(nn.Module):
    def __init__(self,
                n_layers,
                edge_types,
                in_feats_dim,
                hidden_feats_dim,
                out_feats_dim,
                agg,
                drop_out,
                activation
                ):
        super().__init__()
        """GCMC Encoder block for STAR-GCN

        edge_types : list
            all edge types
        in_feats_dim : int
            dimension of input features
        out_feats_dim : int
            dimension of hidden features (output dimension)
        agg : str
            aggreration type
        activation : str
            activation function
        """
        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(GCMCLayer(edge_types = edge_types,
                                            user_feats_dim = in_feats_dim,
                                            item_feats_dim = in_feats_dim,
                                            out_feats_dim = hidden_feats_dim,
                                            agg = agg,
                                            drop_out = drop_out,
                                            activation = activation))
            in_feats_dim = hidden_feats_dim

        self.W_h = nn.Linear(hidden_feats_dim, out_feats_dim)

    def forward(self, graph, ufeats, ifeats, ukey = 'user', ikey = 'item'):
        for encoder in self.encoders:
            ufeats, ifeats = encoder(graph, ufeats, ifeats, ukey, ikey)

        ufeats, ifeats = self.W_h(ufeats), self.W_h(ifeats)

        return ufeats, ifeats

class ReconstructionLayer(nn.Module):
    def __init__(self,
                in_feats_dim,
                org_feats_dim,
                activation,
                hidden_feats_dim
                ):
        super().__init__()
        """Decoder layer for STAR-GCN

        Parameters
        ----------
        in_feats_dim : int
            dimension of encoder or previous decoder layer output
        org_feats_dim : int
            dimension of encoder input
        activation : str
            activation type
        hidden_feats_dim : int (optional)
        """

        self.W_1 = nn.Linear(in_feats_dim, hidden_feats_dim)
        self.activation = activation_map[activation]
        self.W_2 = nn.Linear(hidden_feats_dim, org_feats_dim)

    def forward(self, ufeats, ifeats):
        """
        Parameters
        ----------
        feats : torch.FlaotTensor
            encoder or previous decoder output
        """
        n_users = ufeats.shape[0]
        feats = torch.cat([ufeats, ifeats], dim = 0)

        feats = self.W_1(feats)
        feats = self.activation(feats)
        feats = self.W_2(feats)

        ufeats, ifeats = feats[:n_users, :], feats[n_users:, :]

        return ufeats, ifeats

class ReconstructionDecoder(nn.Module):
    def __init__(self,
                n_layers,
                in_feats_dim,
                org_feats_dim,
                activation,
                hidden_feats_dim = None
                ):
        super().__init__()
        """Decoder block for STAR-GCN

        Parameters
        ----------
        n_layers : int
            number of decoder layers
        in_feats_dim : int
            dimension of encoder or previous decoder layer output
        org_feats_dim : int
            dimension of encoder input
        activation : str
            activation type
        hidden_feats_dim : int (optional)
            according to the paper, all hidden sizes are fixed as org_feats_dim
        """
        if hidden_feats_dim is None:
            hidden_feats_dim = org_feats_dim

        self.decoders = nn.ModuleList()
        for _ in range(n_layers):
            self.decoders.append(ReconstructionLayer(in_feats_dim = in_feats_dim,
                                                    org_feats_dim = org_feats_dim,
                                                    activation = activation,
                                                    hidden_feats_dim = hidden_feats_dim))
            in_feats_dim = hidden_feats_dim

    def forward(self, ufeats, ifeats):
        """
        Parameters
        ----------
        feats : torch.FloatTensor
            encoder output
        """

        for decoder in self.decoders:
            ufeats, ifeats = decoder(ufeats, ifeats)

        return ufeats, ifeats

class STARBlock(nn.Module):
    def __init__(self,
                n_layers_en,
                n_layers_de,
                edge_types,
                in_feats_dim = 64,
                hidden_feats_dim = 250,
                out_feats_dim = 75,
                agg = 'sum',
                drop_out = 0.5,
                activation = 'leakey'
                ):
        super().__init__()
        """STAR-GCN Block with GCMC
        """

        self.encoder_block = GCMCEncoder(n_layers = n_layers_en,
                                        edge_types = edge_types,
                                        in_feats_dim = in_feats_dim,
                                        hidden_feats_dim = hidden_feats_dim,
                                        out_feats_dim = out_feats_dim,
                                        agg = agg,
                                        drop_out = drop_out,
                                        activation = activation)
        
        self.decoder_block = ReconstructionDecoder(n_layers = n_layers_de,
                                                    in_feats_dim = out_feats_dim,
                                                    org_feats_dim = in_feats_dim,
                                                    activation = activation)

    def forward(self, graph, ufeats, ifeats, ukey = 'user', ikey = 'item'):
        ufeats_h, ifeats_h = self.encoder_block(graph, ufeats, ifeats, ukey, ikey)
        ufeats_r, ifeats_r = self.decoder_block(ufeats_h, ifeats_h)

        return ufeats_h, ifeats_h, ufeats_r, ifeats_r

class RatingPrediction(nn.Module):
    def __init__(self,
                in_feats_dim,
                out_feats_dim = 64):
        super().__init__()
        """
        Parameters
        ----------
        in_feats_dim : int
            dimension of encoder output
        out_feats_dim
            dimension of W_u, W_v
        """
        self.W_u = nn.Linear(in_feats_dim, out_feats_dim)
        self.W_v = nn.Linear(in_feats_dim, out_feats_dim)

    def forward(self, graph, ufeats, ifeats):
        ufeats = self.W_u(ufeats)
        ifeats = self.W_v(ifeats)

        pass

if __name__ == '__main__':
    from utils import add_degree

    ratings = [1, 2, 3, 4, 5, 6]
    users = torch.tensor([0,0,0,1,1,2,3,4,4,4,2,2]).chunk(len(ratings))
    items = torch.tensor([0,3,5,1,2,4,5,6,0,1,3,5]).chunk(len(ratings))

    graph_data = {}
    for i in range(len(ratings)):
        graph_data[('user', f'{i+1}', 'item')] = (users[i], items[i])
        graph_data[('item', f'reverse-{i+1}', 'user')] = (items[i], users[i])

    g = dgl.heterograph(graph_data)
    add_degree(graph = g, edge_types = ratings)

    n_users, n_items = 5, 7
    in_feats_dim = 32
    ufeats = torch.rand(n_users, in_feats_dim)
    ifeats = torch.rand(n_items, in_feats_dim)

    block = STARBlock(n_layers_en = 3,
                    n_layers_de = 4,
                    edge_types = ratings,
                    in_feats_dim = in_feats_dim,
                    hidden_feats_dim = 128,
                    out_feats_dim = 24,
                    agg = 'sum',
                    drop_out = 0.,
                    activation = 'relu')

    ufeats_h, ifeats_h, ufeats_r, ifeats_r = block(g, ufeats, ifeats)

    print('hidden feautures') # (n_users, out_feats_dim), (n_items, out_feats_dim)
    print(ufeats_h.shape, ifeats_h.shape)

    print('reconstruction') # (n_users, in_feats_dim, n_items, in_feats_dim)
    print(ufeats_r.shape, ifeats_r.shape)