import torch
import numpy as np
from torch.nn import functional as F

def identity_mapping(x):
    return x

activation_map = {
    'relu' : F.relu,
    'leaky' : F.leaky_relu,
    'selu' : F.selu,
    'sigmoid' : F.sigmoid,
    'tanh' : F.tanh,
    'none' : identity_mapping
}

def add_degree(graph, edge_types, symmetric = True, n_users = None, n_items = None):
    def _calc_norm(x):
        x = x.numpy().astype('float32')
        x[x == 0.] = np.inf
        x = torch.FloatTensor(1. / np.sqrt(x))

        return x.unsqueeze(1)

    user_ci = []
    user_cj = []
    movie_ci = []
    movie_cj = []
    for r in edge_types:
        user_ci.append(graph[f'reverse-{r}'].in_degrees())
        movie_ci.append(graph[f'{r}'].in_degrees())
        
        if symmetric:
            user_cj.append(graph[f'{r}'].out_degrees())
            movie_cj.append(graph[f'reverse-{r}'].out_degrees())

    user_ci = _calc_norm(sum(user_ci))
    movie_ci = _calc_norm(sum(movie_ci))

    if symmetric:
        user_cj = _calc_norm(sum(user_cj))
        movie_cj = _calc_norm(sum(movie_cj))
    else:
        user_cj = torch.ones((n_users,))
        movie_cj = torch.ones((n_items,))

    graph.nodes['user'].data.update({'ci': user_ci, 'cj': user_cj})
    graph.nodes['item'].data.update({'ci': movie_ci, 'cj': movie_cj})