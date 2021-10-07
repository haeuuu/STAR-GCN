import torch
import torch.nn as nn

from utils import activation_map


class ExternalFeatures(nn.Module):
    def __init__(self,
                feats,
                out_feats_dim,
                activation
                ):
        super().__init__()
        """Two layer feedforward NN for external feature transformation

        Parameters
        ----------
        feats : torch.FloatTensor
            external features
        iout_feats_dim : int
            dimension of output feature
        activation : str
            activation type
        """
        self.feats = feats

        self.W_1 = nn.Linear(feats.shape[1], out_feats_dim)
        self.activation = activation_map[activation]
        self.W_2 = nn.Linear(out_feats_dim, out_feats_dim)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.W_1.weight)
        torch.nn.init.xavier_normal_(self.W_2.weight)

    def transform(self, feats):
        feats = self.W_1(feats)
        feats = self.activation(feats)
        feats = self.W_2(feats)

        return feats

    def forward(self, idx):
        """
        Parameters
        ----------
        idx : list or torch.LongTensor
            target indices
        
        Returns
        -------
        feats : torch.Tensor
            W_{2} \sigma( W_{1} @ feats)
        """
        feats = self.transform(self.feats[idx])

        return feats

class InputFeatures(nn.Module):
    def __init__(self,
                n_nodes,
                emb_dim,
                p_zero = 0.2,
                p_freeze = 0.,
                efeats = None,
                efeats_dim = None,
                activation = None
                ):
        super().__init__()
        """STAR-GCN input features

        Parameters
        ----------
        n_nodes : int
            number of nodes
        emb_dim : int
            embedding size
        """
        self.p_zero = p_zero
        self.p_freeze = p_freeze

        if efeats is None:
            self.external_feats = None
        else:
            self.external_feats = ExternalFeatures(feats = efeats,
                                                    out_feats_dim = efeats_dim,
                                                    activation = activation)
        self.emb_dim = emb_dim
        self.feats = nn.Embedding(n_nodes, emb_dim)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.feats.weight)

    def get_unseen_feature(self, idx, efeats = None):
        """generate unseen node features to inductive inference

        Parameters
        ----------
        efeats : torch.FloatTensor (optional)
            externel features
        """
        device = self.feats.weight.device

        feats = torch.zeros(len(idx), self.emb_dim).to(device)
        if efeats is not None:
            efeats = self.external_feats.transform(efeats.to(device))
            feats = torch.cat([feats, efeats], dim = -1)

        return feats

    def generate_mask(self, idx, p):
        length, device = len(idx), idx.device
        if p <= 0.:
            mask = None
        else:
            mask = (torch.rand(length, ) < p).to(device)

        return mask

    def forward(self, idx, masked = True):
        feats = self.feats(idx)

        if masked:
            mask_zero = self.generate_mask(idx, self.p_zero)
            mask_freeze = self.generate_mask(idx, self.p_freeze)
            if mask_zero is not None:
                feats.masked_fill_(mask_zero.unsqueeze(1), 0.)
        else:
            mask_zero, mask_freeze = None, None

        if self.external_feats is not None:
            external_feats = self.external_feats(idx)
            feats = torch.cat([feats, external_feats], dim = -1)

        return feats, mask_zero, mask_freeze