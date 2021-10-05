import torch
import torch.nn as nn

import dgl
import dgl.function as fn


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

    def forward(self, graph, ufeats, ifeats, ukey = 'user', ikey = 'item'):
        ufeats = self.W_u(ufeats)
        ifeats = self.W_v(ifeats)

        with graph.local_scope():
            graph.nodes[ukey].data['h'] = ufeats
            graph.nodes[ikey].data['h'] = ifeats
            graph.apply_edges(fn.u_dot_v('h', 'h', 'r'))
            pred = graph.edata['r']

        return pred

class RatingPredictionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction = 'mean')

    def forward(self, ground_truth, prediction):
        """
        Parameters
        ----------
        ground_truth, prediction
            shape : (n_edges, )
        """
        loss = self.mse(ground_truth.float(), prediction)

        return loss

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction = 'sum')

    def forward(self, ufeats, ifeats, ufeats_r, ifeats_r):
        loss_u = self.mse(ufeats, ufeats_r) / ufeats.shape[0]
        loss_v = self.mse(ifeats, ifeats_r) / ifeats.shape[0]

        return (loss_u + loss_v)/2

class Criterion(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
        self.rating_loss = RatingPredictionLoss()
        self.reconstruction_loss = ReconstructionLoss()

    def forward(self, ground_truth, prediction, ufeats, ifeats, ufeats_r, ifeats_r):
        loss_t = self.rating_loss(ground_truth, prediction)
        loss_r = self.reconstruction_loss(ufeats, ifeats, ufeats_r, ifeats_r)

        return loss_t + self.weight * loss_r