import torch
import torch.nn as nn

import dgl
import dgl.function as fn

from blocks import STARBlock
from loss import RatingPrediction


class STARGCN(nn.Module):
    def __init__(self,
                n_blocks,
                n_layers_en,
                n_layers_de,
                recurrent,
                edge_types,
                in_feats_dim = 64,
                en_hidden_feats_dim = 250,
                r_hidden_feats_dim = 64,
                out_feats_dim = 75,
                agg = 'sum',
                drop_out = 0.5,
                activation = 'leaky',
                ):
        super().__init__()
        """STAR-GCN for link prediction (https://arxiv.org/pdf/1905.13129.pdf)

        Parameters
        ----------
        n_blocks : int
            number of STARBlock module
        recurrent : bool
            whether to share weights among STARBlocks
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

        self.rating_prediction = RatingPrediction(in_feats_dim = out_feats_dim,
                                                out_feats_dim = r_hidden_feats_dim)

    def forward(self, enc_graph, dec_graph, ufeats, ifeats, ufreeze = None, ifreeze = None, ukey = 'user', ikey = 'item'):
        """
        Parameters
        ----------
        enc_graph : dgl.heterograph
            graph for encoder.forward pass
        dec_graph : dgl.homograph
            graph for rating_prediction.forward pass

        Returns
        -------
        all_ratings : list of torch.FloatTensor
            len = n_blocks
            list of rating prediction results

        all_recom_feats : list of torch.FloatTensor
            len = n_blocks
            list of reconstructed features
        """

        all_ratings, all_recon_feats = [], []
        for i in range(self.n_blocks):
            block = self.blocks[i] if not self.recurrent else self.blocks[0]
            ufeats_h, ifeats_h, ufeats_r, ifeats_r = block(enc_graph, ufeats, ifeats, ufreeze, ifreeze, ukey, ikey)
            pred_ratings = self.rating_prediction(dec_graph, ufeats_h, ifeats_h).squeeze(1)
            
            all_ratings.append(pred_ratings)
            all_recon_feats.append((ufeats_r, ifeats_r))

        return all_ratings, all_recon_feats