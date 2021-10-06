import sys
sys.path.append('..')

import os
import time
import fire

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import STARGCN
from data import MovieLens

from feature import InputFeatures
from loss import RatingPredictionLoss, Criterion


class Trainer:
    def __init__(self,
                data_name = 'ml-100k',
                valid_ratio = 0.1,
                test_ratio = 0.1
                ):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = MovieLens(name = data_name,
                                test_ratio = test_ratio,
                                valid_ratio = valid_ratio,
                                device = self.device)

        self.dataset.train_enc_graph = self.dataset.train_enc_graph.int().to(self.device)
        self.dataset.train_dec_graph = self.dataset.train_dec_graph.int().to(self.device)
        self.dataset.valid_enc_graph = self.dataset.train_enc_graph
        self.dataset.valid_dec_graph = self.dataset.valid_dec_graph.int().to(self.device)
        self.dataset.test_enc_graph = self.dataset.test_enc_graph.int().to(self.device)
        self.dataset.test_dec_graph = self.dataset.test_dec_graph.int().to(self.device)

    def train(self,
            n_blocks = 2,
            n_layers_en = 1,
            n_layers_de = 1,
            recurrent = True,
            in_feats_dim = 32,
            efeats_dim = None,
            en_hidden_feats_dim = 250,
            out_feats_dim = 75,
            r_hidden_feats_dim = 64,
            agg = 'sum',
            drop_out = 0.5,
            activation = 'leaky',
            weight = 0.1,
            p_zero = 0.0,
            p_freeze = 0.0,
            lr = 0.002,
            iteration = 1000,
            log_interval = 1,
            early_stopping = 150,
            lr_intialize_step = 100,
            lr_decay = 0.5,
            train_min_lr = 0.0005
            ):

        n_users, n_items = self.dataset.user_feature.shape[0], self.dataset.movie_feature.shape[0]

        if efeats_dim is None:
            user_features = InputFeatures(n_nodes = n_users,
                                        emb_dim = in_feats_dim,
                                        p_zero = p_zero / 2,
                                        p_freeze = p_freeze / 2)
            movie_features = InputFeatures(n_nodes = n_items,
                                        emb_dim = in_feats_dim,
                                        p_zero = p_zero / 2,
                                        p_freeze = p_freeze / 2)
        else:
            user_features = InputFeatures(n_nodes = n_users,
                                        emb_dim = in_feats_dim,
                                        p_zero = p_zero / 2,
                                        p_freeze = p_freeze / 2,
                                        efeats = self.dataset.user_feature,
                                        efeats_dim = efeats_dim,
                                        activation = activation)

            movie_features = InputFeatures(n_nodes = n_items,
                                        emb_dim = in_feats_dim,
                                        p_zero = p_zero / 2,
                                        p_freeze = p_freeze / 2,
                                        efeats = self.dataset.movie_feature,
                                        efeats_dim = efeats_dim,
                                        activation = activation)

            in_feats_dim += efeats_dim

        model = STARGCN(n_blocks = n_blocks,
                        n_layers_en = n_layers_en,
                        n_layers_de = n_layers_de,
                        recurrent = recurrent,
                        edge_types = self.dataset.possible_rating_values,
                        in_feats_dim = in_feats_dim,
                        en_hidden_feats_dim = en_hidden_feats_dim,
                        r_hidden_feats_dim = r_hidden_feats_dim,
                        out_feats_dim = out_feats_dim,
                        agg = agg,
                        drop_out = drop_out,
                        activation = activation)

        print(model)
        print(user_features)
        print(movie_features)

        device = self.device
        model = model.to(device)
        user_features = user_features.to(device)
        movie_features = movie_features.to(device)

        criterion = Criterion(weight = weight)

        params = list(model.parameters()) + \
            list(user_features.parameters()) + list(movie_features.parameters())
        optimizer = optim.Adam(params, lr = lr)

        train_gt_ratings = self.dataset.train_truths

        best_valid_rmse = np.inf
        no_better_valid = 0
        best_iter = -1
        count_loss = 0

        print(f"Start training on {device}...")
        for iter_idx in range(iteration):
            model.train()

            # TODO : implement inductive version and masked learning
            ufeats, umask_zero, umask_freeze = user_features(torch.arange(n_users).to(device))
            ifeats, imask_zero, imask_freeze = movie_features(torch.arange(n_items).to(device))

            all_ratings, all_recon_feats = \
                model(self.dataset.train_enc_graph, self.dataset.train_dec_graph, ufeats, ifeats)
                    
            loss, rmse = 0., 0.
            for pred_ratings, (ufeats_r, ifeats_r) in zip(all_ratings, all_recon_feats):
                _rmse, _loss = criterion(train_gt_ratings,
                                        pred_ratings,
                                        ufeats,
                                        ifeats,
                                        ufeats_r,
                                        ifeats_r)
                rmse += _rmse
                loss += _loss

            rmse /= len(all_ratings)

            count_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            if iter_idx and iter_idx % log_interval == 0:
                log = f"[{iter_idx}/{iteration}-iter] | [train] loss : {count_loss/iter_idx:.4f}, rmse : {rmse:.4f}"
                count_rmse, count_num = 0, 0

            if iter_idx and iter_idx % (log_interval*10) == 0:
                valid_rmse = self.evaluate(model, n_users, n_items, user_features, movie_features, data_type = 'valid')
                log += f" | [valid] rmse : {valid_rmse:.4f}"

                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    no_better_valid = 0
                    best_iter = iter_idx
                    best_test_rmse = self.evaluate(model, n_users, n_items, user_features, movie_features, data_type = 'test')
                    log += f" | [test] rmse : {best_test_rmse:.4f}"

                    torch.save(model, './model.pt')

                else:
                    no_better_valid += 1
                    if no_better_valid > early_stopping:
                        print("Early stopping threshold reached. Stop training.")
                        break
                    if no_better_valid > lr_intialize_step:
                        new_lr = max(lr * lr_decay, train_min_lr)
                        if new_lr < lr:
                            lr = new_lr
                            print("\tChange the LR to %g" % new_lr)
                            for p in optimizer.param_groups:
                                p['lr'] = lr
                            no_better_valid = 0

            if iter_idx and iter_idx  % log_interval == 0:
                print(log)

        print(f'[END] Best Iter : {best_iter} Best Valid RMSE : {best_valid_rmse:.4f}, Best Test RMSE : {best_test_rmse:.4f}')

    def evaluate(self, model, n_users, n_items, user_features, movie_features, data_type = 'valid'):
        if data_type == "valid":
            gt_ratings = self.dataset.valid_truths
            dec_graph = self.dataset.valid_dec_graph
        elif data_type == "test":
            gt_ratings = self.dataset.test_truths
            dec_graph = self.dataset.test_dec_graph

        model.eval()
        with torch.no_grad():
            get_rmse = RatingPredictionLoss()

            # TODO : inductive version inference
            
            # ufeats = user_features.get_unseen_feature(efeats = self.dataset.user_feature)
            # ifeats = movie_features.get_unseen_feature(efeats = self.dataset.movie_feature)

            # ufeats = user_features.get_unseen_feature(torch.arange(n_users).to(self.device))
            # ifeats = movie_features.get_unseen_feature(torch.arange(n_items).to(self.device))

            ufeats, _, _ = user_features(torch.arange(n_users).to(self.device))
            ifeats, _, _ = movie_features(torch.arange(n_items).to(self.device))

            all_ratings, _ = model(self.dataset.train_enc_graph, dec_graph, ufeats, ifeats)
            rmse = 0.
            for pred_ratings in all_ratings:
                rmse += get_rmse(gt_ratings, pred_ratings).pow(1/2)

            rmse /= len(all_ratings)
            
        return rmse

if __name__ == '__main__':
    SEED = 152
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    fire.Fire(Trainer)
