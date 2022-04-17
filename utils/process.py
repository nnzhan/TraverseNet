import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import pandas as pd
import torch
import tqdm

def process_t_graph(srclist, tgtlist, dist, T, num_nodes, window=12):
    #assign the same edge type to connections between a node's current state and its neigbhor's all historial states.
    df = pd.DataFrame({'src': srclist, 'tgt': tgtlist, 'dist': dist})
    df = df.sort_values(by=['tgt','dist'], ignore_index=True)
    print(df.loc[0:30])
    dfn = pd.DataFrame(columns=['rel', 'src', 'src_t', 'tgt', 'tgt_t', 'dis'])
    rel = dict()
    for i in tqdm.trange(len(df)):
        if df.loc[i][1] not in rel.keys():
            ctr = 0
            rel[df.loc[i][1]] = [str(ctr)]
        else:
            ctr += 1
        dis = df.loc[i][2]
        if dis==0:
            rela = str(ctr)+"_-1"
            ew = 1
        else:
            rela = str(ctr)+"_1"
            ew = dis

        for j in range(T):
            s = j-window
            if s<0:
                s=0
            for k in range(s,j+1):
                dfn = dfn.append({'rel': rela, 'src': int(df.loc[i][0]), 'src_t': int(k), 'tgt': int(df.loc[i][1]), 'tgt_t': int(j), 'dis': ew},
                                 ignore_index=True)
            if j ==0 and df.loc[i][0] == df.loc[i][1]: 
                continue
            #if df.loc[i][0] == df.loc[i][1]:
            dfn = dfn.append({'rel': rela, 'src': int(df.loc[i][1]), 'src_t': int(j), 'tgt': int(df.loc[i][1]), 'tgt_t': int(j), 'dis': ew},
                                 ignore_index=True)
    print(dfn)
    g = dict()
    ds = dict()
    for i in tqdm.trange(len(dfn)):
        if dfn.loc[i][0] not in g.keys():
            g[dfn.loc[i][0]] = pd.DataFrame(columns=['rel', 'src', 'tgt'])
            ds[dfn.loc[i][0]] = []
        src = dfn.loc[i][2] * num_nodes + dfn.loc[i][1]
        tgt = dfn.loc[i][4] * num_nodes + dfn.loc[i][3]
        g[dfn.loc[i][0]] = g[dfn.loc[i][0]].append({'rel': dfn.loc[i][0], 'src': src, 'tgt': tgt}, ignore_index=True)
        ds[dfn.loc[i][0]].append(dfn.loc[i][5])

    graph_data = dict()
    print('number of relations ', len(g.keys()))
    for k in g.keys():
        key = ('v', k, 'v')
        value = (torch.tensor(g[k]['src'].to_numpy(dtype=int)), torch.tensor(g[k]['tgt'].to_numpy(dtype=int)))
        graph_data[key] = value
        ds[k] = torch.tensor(ds[k])
    # g = dgl.heterograph(graph_data)
    return graph_data, ds


def process_st_graph(srclist, tgtlist, dist, T, num_nodes, window=12):
    #construct a convention st graph.
    df = pd.DataFrame({'src': srclist, 'tgt': tgtlist, 'dist': dist})
    df = df.sort_values(by=['tgt','dist'], ignore_index=True)
    dfn = pd.DataFrame(columns=['rel', 'src', 'src_t', 'tgt', 'tgt_t', 'dis'])
    print(df.loc[0:30])
    for i in tqdm.trange(len(df)):
        src = int(df.loc[i][0])
        tgt = int(df.loc[i][1])
        for t in range(T):
            dfn = dfn.append({'rel': '1', 'src': src, 'src_t': t, 'tgt': tgt, 'tgt_t': t, 'dis': df.loc[i][2]},
                             ignore_index=True)
        # if src==tgt:
        #     for j in range(T):
        #         s = j - window
        #         if s < 0:
        #             s = 0
        #         for k in range(s,j+1):
        #             dfn = dfn.append({'rel': '0', 'src': src, 'src_t': int(k), 'tgt': tgt,
        #                               'tgt_t': int(j), 'dis':0}, ignore_index=True)
    print(dfn)
    g = dict()
    ds = dict()
    for i in tqdm.trange(len(dfn)):
        if dfn.loc[i][0] not in g.keys():
            g[dfn.loc[i][0]] = pd.DataFrame(columns=['rel', 'src', 'tgt'])
            ds[dfn.loc[i][0]] = []
        src = dfn.loc[i][2] * num_nodes + dfn.loc[i][1]
        tgt = dfn.loc[i][4] * num_nodes + dfn.loc[i][3]
        g[dfn.loc[i][0]] = g[dfn.loc[i][0]].append({'rel': dfn.loc[i][0], 'src': src, 'tgt': tgt}, ignore_index=True)
        ds[dfn.loc[i][0]].append(dfn.loc[i][5])

    graph_data = dict()
    print('number of relations ', len(g.keys()))
    for k in g.keys():
        key = ('v', k, 'v')
        value = (torch.tensor(g[k]['src'].to_numpy(dtype=int)), torch.tensor(g[k]['tgt'].to_numpy(dtype=int)))
        graph_data[key] = value
        ds[k] = torch.tensor(ds[k])
    # g = dgl.heterograph(graph_data)
    return graph_data, ds


def process_f_graph(srclist, tgtlist, dist, T, num_nodes, window=12):
    #to do: a function to be deleted
    df = pd.DataFrame({'src': srclist, 'tgt': tgtlist, 'dist': dist})
    df = df.sort_values(by=['tgt','dist'], ignore_index=True)
    print(df.loc[0:30])
    dfn = pd.DataFrame(columns=['rel', 'src', 'src_t', 'tgt', 'tgt_t', 'dis'])
    rel = dict()
    for i in range(len(df)):
        print(i)
        if df.loc[i][1] not in rel.keys():
            ctr = 0
            rel[df.loc[i][1]] = [str(ctr)]
        else:
            ctr += 1
        dis = df.loc[i][2]
        if dis==0:
            rela = str(ctr)+"_-1"
            ew = 1
        elif dis>1e8:
            rela = str(ctr)+"_0"
            ew = 1e-9 - dis
        else:
            rela = str(ctr)+"_1"
            ew = dis

        for j in range(T):
            for k in range(T):
                dfn = dfn.append({'rel': rela, 'src': int(df.loc[i][0]), 'src_t': int(k), 'tgt': int(df.loc[i][1]), 'tgt_t': int(j), 'dis': ew},
                                 ignore_index=True)
    print(dfn)
    g = dict()
    ds = dict()
    for i in range(len(dfn)):
        if dfn.loc[i][0] not in g.keys():
            g[dfn.loc[i][0]] = pd.DataFrame(columns=['rel', 'src', 'tgt'])
            ds[dfn.loc[i][0]] = []
        src = dfn.loc[i][2] * num_nodes + dfn.loc[i][1]
        tgt = dfn.loc[i][4] * num_nodes + dfn.loc[i][3]
        g[dfn.loc[i][0]] = g[dfn.loc[i][0]].append({'rel': dfn.loc[i][0], 'src': src, 'tgt': tgt}, ignore_index=True)
        ds[dfn.loc[i][0]].append(dfn.loc[i][5])

    graph_data = dict()
    print('number of relations ', len(g.keys()))
    for k in g.keys():
        key = ('v', k, 'v')
        value = (torch.tensor(g[k]['src'].to_numpy(dtype=int)), torch.tensor(g[k]['tgt'].to_numpy(dtype=int)))
        graph_data[key] = value
        ds[k] = torch.tensor(ds[k])
    # g = dgl.heterograph(graph_data)
    return graph_data, ds



def process_s_graph(srclist, tgtlist, dist, T, num_nodes, window=12):
    # assign the same edge type to connections between a node's current state and all of its neigbhors' current states.
    df = pd.DataFrame({'src': srclist, 'tgt': tgtlist, 'dist':dist})
    df = df.sort_values(by=['tgt', 'dist'], ignore_index=True)
    print(df.loc[0:30])
    dfn = pd.DataFrame(columns=['rel', 'src', 'src_t', 'tgt', 'tgt_t'])
    for i in tqdm.trange(len(df)):
        dis = df.loc[i][2]
        if dis==0:
            rela = "_-1"
        elif dis>1e8:
            rela = "_0"
        else:
            rela = "_1"
        for j in range(T):
            s = j-window
            if s<0:
                s=0
            for k in range(s,j+1):
                dfn = dfn.append({'rel': str(j-k)+rela, 'src': int(df.loc[i][0]), 'src_t': int(k), 'tgt': int(df.loc[i][1]), 'tgt_t': int(j)},
                                 ignore_index=True)
    print(dfn)
    g = dict()
    for i in tqdm.trange(len(dfn)):
        if dfn.loc[i][0] not in g.keys():
            g[dfn.loc[i][0]] = pd.DataFrame(columns=['rel', 'src', 'tgt'])
        src = dfn.loc[i][2] * num_nodes + dfn.loc[i][1]
        tgt = dfn.loc[i][4] * num_nodes + dfn.loc[i][3]
        g[dfn.loc[i][0]] = g[dfn.loc[i][0]].append({'rel': dfn.loc[i][0], 'src': src, 'tgt': tgt}, ignore_index=True)
    graph_data = dict()
    print('number of relations ', len(g.keys()))
    for k in g.keys():
        key = ('v', 'r' + str(k), 'v')
        value = (torch.tensor(g[k]['src'].to_numpy(dtype=int)), torch.tensor(g[k]['tgt'].to_numpy(dtype=int)))
        graph_data[key] = value
    # g = dgl.heterograph(graph_data)
    return graph_data



def process_a_graph(srclist, tgtlist, T, num_nodes, window=12):
    # to do: a function to be deleted.
    df = pd.DataFrame({'src': srclist, 'tgt': tgtlist})
    df = df.sort_values(by=['tgt', 'src'], ignore_index=True)
    print(df)
    dfn = pd.DataFrame(columns=['rel', 'src', 'src_t', 'tgt', 'tgt_t'])
    for i in range(len(df)):
        print(i)
        for j in range(T):
            s = j-window
            if s<0:
                s=0
            for k in range(s,j+1):
                dfn = dfn.append({'rel': 0, 'src': df.loc[i][0], 'src_t': k, 'tgt': df.loc[i][1], 'tgt_t': j},
                                 ignore_index=True)
    print(dfn)
    g = dict()
    for i in range(len(dfn)):
        if dfn.loc[i][0] not in g.keys():
            g[dfn.loc[i][0]] = pd.DataFrame(columns=['rel', 'src', 'tgt'])
        src = dfn.loc[i][2] * num_nodes + dfn.loc[i][1]
        tgt = dfn.loc[i][4] * num_nodes + dfn.loc[i][3]
        g[dfn.loc[i][0]] = g[dfn.loc[i][0]].append({'rel': dfn.loc[i][0], 'src': src, 'tgt': tgt}, ignore_index=True)
    graph_data = dict()
    print('number of relations ', len(g.keys()))
    for k in g.keys():
        key = ('v', 'r' + str(k), 'v')
        value = (torch.tensor(g[k]['src'].to_numpy(dtype=int)), torch.tensor(g[k]['tgt'].to_numpy(dtype=int)))
        graph_data[key] = value
    # g = dgl.heterograph(graph_data)
    return graph_data



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32)

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32)

def trans_adj(adj):
    adj = sp.coo_matrix(adj)
    colsum = np.array(adj.sum(0)).flatten()
    d_inv = np.power(colsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return adj.dot(d_mat).astype(np.float32)

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = linalg.eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials
