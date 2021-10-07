import torch
import torch.nn as nn

import dgl
import dgl.function as fn

from blocks import STARBlock


class STARGCN(nn.Module):
    def __init__(self,
                n_blocks,
                n_layers_en,
                n_layers_de,
                recurrent,
                edge_types,
                in_feats_dim = 64,
                en_hidden_feats_dim = 250,
                out_feats_dim = 75,
                agg = 'sum',
                drop_out = 0.5,
                activation = 'leaky',
                ):
        super().__init__()
        """
        en_hidden_feats_dim : int
            dimension of GCMC transformation
        r_hidden_feats_dim : int
            dimension of rating prediction's transformation
        """

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(STARBlock(n_layers_en = n_layers_en,
                                        n_layers_de = n_layers_de,
                                        edge_types = edge_types,
                                        in_feats_dim = in_feats_dim,
                                        hidden_feats_dim = en_hidden_feats_dim,
                                        out_feats_dim = out_feats_dim,
                                        agg = agg,
                                        drop_out = drop_out,
                                        activation = activation))
            if recurrent:
                break

        self.n_blocks = n_blocks
        self.recurrent = recurrent

    def forward(self, graph, ufeats, ifeats, ukey = 'user', ikey = 'item'):

        results = []

        for block in self.blocks:
            ufeats_h, ifeats_h, ufeats_r, ifeats_r = block(graph, ufeats, ifeats, ukey, ikey)
            results.append((ufeats_h, ifeats_h, ufeats_r, ifeats_r))

        return results