import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torch
import copy
from torch import nn
from layers.gat_layer import GATConvs
from dgl.ops import edge_softmax
from layers.smt import HeteroGraphConv
def module_list(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


class Norm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.size = dim
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class Norm1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d((dim))
        self.dim = dim
    def forward(self, x):
        sh = x.shape
        x = x.view(-1, self.dim, sh[1], sh[2])
        nx = self.norm(x)
        nx = nx.view(-1,sh[1],sh[2])
        return nx


def my_agg_func(tensors, dsttype):
    return tensors[0]

def my_agg_func1(tensors, dsttype):
    return tensors[-1]

class EncoderLayer(nn.Module):
    def __init__(self, num_nodes, dim, heads, rel_keys, dropout, layer_id, bi=True):
        super().__init__()
        self.norm_1 = Norm1(num_nodes)
        self.norm_2 = Norm1(num_nodes)
        block = dict()
        if bi:
            self.attn1 = GATConvs(dim, dim // heads, heads, num_nodes,layer_id)
            self.attn2 = GATConvs(dim, dim // heads, heads, num_nodes,layer_id)

            for it in rel_keys:
                k = it[1]
                s = k.split('_')
                if len(s)!=1:
                    if s[1] == '-1':
                        block[k] = self.attn1
                    else:
                        block[k] = self.attn2
                else:
                    if s[0] == '0':
                        block[k] = self.attn1
                    else:
                        block[k] = self.attn2

                # if it[1]=='0':
                #     block[it[1]] = self.attn1
                # elif it[1]=='1':
                #     block[it[1]] = self.attn2
                # else:
                #     block[it[1]] = self.attn3
                # k = it[1]
                # s = k.split('_')
                # if s[1] == '-1':
                #     block[k] = GATConv(dim, dim // heads, heads, num_nodes)
                # elif s[1] == '0':
                #     block[k] = self.attn1
                # else:
                #     block[k] = self.attn2
        else:
            for it in rel_keys:
                k = it[1]
                block[k] = GATConvs(dim, dim // heads, heads, num_nodes)


        self.conv = HeteroGraphConv(block,dim,len(rel_keys))
        #self.conv = dglnn.HeteroGraphConv(block,aggregate='mean')
        self.ff = FeedForward(dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)


        #self.tp = nn.Conv2d(in_channels=18, out_channels=1, kernel_size=(1, 1))
    def forward(self, g, x):
        # x NT*batch_size*dim
        h = {'v': x['v']}
        h['v'] = self.norm_1(h['v'])
        h = self.conv(g, h)
        shape = h['v'].shape
        h['v'] = h['v'].reshape(*shape[:-2], -1)
        #h['v'] = self.tp(h['v']).squeeze()
        x['v'] = x['v'] + self.dropout_1(h['v'])
        h['v'] = self.norm_2(x['v'])
        h['v'] = x['v'] + self.dropout_2(self.ff(h['v']))
        return h


class Encoder(nn.Module):
    def __init__(self, num_nodes, dim, heads, relkeys, num_layer, dropout):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layer = num_layer
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(EncoderLayer(num_nodes, dim, heads, relkeys, dropout, i))
        self.norm = Norm1(num_nodes)
        self.dim = dim


    def forward(self, g, x):
        for i in range(self.num_layer):
            x = self.layers[i](g, x)
        x['v'] = self.norm(x['v'])
        return x


class Encoder1(nn.Module):
    def __init__(self, num_nodes, dim, heads, relkeys1, relkeys2, num_layer, dropout):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layer = num_layer
        self.layers1 = module_list(EncoderLayer(num_nodes, dim, heads, relkeys1, dropout, 2), num_layer)
        self.layers2 = module_list(EncoderLayer(num_nodes, dim, heads, relkeys2, dropout, 2), num_layer)
        self.convl =  module_list(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(1, 3)),num_layer)
        self.norm = Norm1(num_nodes)
        self.dim = dim

    def forward(self, g1, g2, x):
        for i in range(self.num_layer):
            x = self.layers1[i](g1, x)
            x = self.layers2[i](g2, x)
        x['v'] = self.norm(x['v'])
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, d_ff=128, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, dim)

    def forward(self, x):
        x = self.dropout(F.gelu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


