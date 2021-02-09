import torch
import math
from torch import nn
import torch.nn.functional as F
from module.traversebody import Encoder, Encoder1
import numpy as np
from torch.utils.checkpoint import checkpoint
class Norm1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(dim)
        self.dim = dim
    def forward(self, x):
        sh = x.shape
        x = x.transpose(2,1)
        x = x.view(-1, self.dim, sh[1], sh[2])
        nx = self.norm(x)
        nx = nx.transpose(1,2)
        return nx

class PrePreplayer(nn.Module):
    def __init__(self, in_dim, dim, num_nodes, seq_l, dropout):
        super().__init__()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=dim, kernel_size=(1, 1))
        self.norm1 = torch.nn.LayerNorm((dim,num_nodes,seq_l))
        self.dropout = dropout

    def forward(self,x, dummy):
        h = self.start_conv(x)
        h = self.norm1(h)
        return h

class PostPreplayer(nn.Module):
    def __init__(self, dim, out_dim, num_nodes, seq_l, dropout):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm((dim,num_nodes,seq_l))
        self.end_conv_1 = nn.Conv2d(in_channels=dim, out_channels=out_dim**2, kernel_size=(1, seq_l))
        self.end_conv_2 = nn.Conv2d(in_channels=out_dim**2, out_channels=out_dim, kernel_size=(1, 1))
        self.dim = dim
        self.seq_l = seq_l
        self.num_nodes = num_nodes
        self.dropout = dropout
    def forward(self, x):
        h = self.norm1(x)
        h = F.relu(self.end_conv_1(h))
        h = self.end_conv_2(h)
        return h

class TraverseNet(nn.Module):
    def __init__(self, net_params, g1, relkeys):
        super().__init__()
        self.start_conv = PrePreplayer(net_params['in_dim'], net_params['dim'], net_params['num_nodes'], net_params['seq_in_len'], net_params['dropout'])
        self.end_conv = PostPreplayer(net_params['dim'], net_params['seq_out_len'], net_params['num_nodes'], net_params['seq_in_len'], net_params['dropout'])
        self.transformer = Encoder(net_params['num_nodes'], net_params['dim'], net_params['heads'], relkeys, net_params['num_layers'], net_params['dropout'])

        self.in_dim = net_params['in_dim']
        self.num_nodes = net_params['num_nodes']
        self.seq_in_len = net_params['seq_in_len']
        self.seq_out_len = net_params['seq_out_len']

        self.dim = net_params['dim']
        self.g1 = g1
        self.cl_decay_steps = net_params['cl_decay_steps']
        self.num_layer = net_params['num_layers']

    def _init_pos(self,sq,dim):
        enc = torch.Tensor(sq,dim)
        for t in range(sq):
            for i in range(0, dim, 2):
                enc[t, i] = math.sin(t / (10000 ** ((2 * i)/dim)))
                enc[t, i + 1] = math.cos(t / (10000 ** ((2 * (i + 1))/dim)))
        return enc
    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, src, dummy):
        x = src[:,:self.in_dim,:,-(self.seq_in_len):]
        h = self.start_conv(x, dummy)
        h = h.permute(3, 2, 0, 1)
        h = h.reshape(self.num_nodes * self.seq_in_len, -1, self.dim)
        hx = {'v': h}
        out = self.transformer(self.g1, hx)
        out = out['v'].reshape(self.seq_in_len, self.num_nodes, -1, self.dim)
        out = out.permute(2, 3, 1, 0)
        out = self.end_conv(out)
        return out

#interleave spatial attentions with temporal attentions
class TraverseNetst(nn.Module):
    def __init__(self, net_params, g1, g2, relkeys1, relkeys2):
        super().__init__()
        self.start_conv = PrePreplayer(net_params['in_dim'], net_params['dim'], net_params['num_nodes'], net_params['seq_in_len'], net_params['dropout'])
        self.end_conv = PostPreplayer(net_params['dim'], net_params['seq_out_len'], net_params['num_nodes'], net_params['seq_in_len'], net_params['dropout'])
        self.transformer = Encoder1(net_params['num_nodes'], net_params['dim'], net_params['heads'], relkeys1, relkeys2, net_params['num_layers'], net_params['dropout'])

        self.in_dim = net_params['in_dim']
        self.num_nodes = net_params['num_nodes']
        self.seq_in_len = net_params['seq_in_len']
        self.seq_out_len = net_params['seq_out_len']

        self.dim = net_params['dim']
        self.g1 = g1
        self.g2 = g2
        self.cl_decay_steps = net_params['cl_decay_steps']
        self.num_layer = net_params['num_layers']

    def _init_pos(self,sq,dim):
        enc = torch.Tensor(sq,dim)
        for t in range(sq):
            for i in range(0, dim, 2):
                enc[t, i] = math.sin(t / (10000 ** ((2 * i)/dim)))
                enc[t, i + 1] = math.cos(t / (10000 ** ((2 * (i + 1))/dim)))
        return enc
    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, src, dummy):
        x = src[:,:self.in_dim,:,-(self.seq_in_len):]
        h = self.start_conv(x, dummy)
        h = h.permute(3, 2, 0, 1)
        h = h.reshape(self.num_nodes * self.seq_in_len, -1, self.dim)
        hx = {'v': h}
        out = self.transformer(self.g1, self.g2, hx)
        out = out['v'].reshape(self.seq_in_len, self.num_nodes, -1, self.dim)
        out = out.permute(2, 3, 1, 0)
        out = self.end_conv(out)
        return out
