# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.astgcn_block import ASTGCN_block


class ASTGCNnet(nn.Module):

    def __init__(self, supports, net_params, DEVICE):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        super(ASTGCNnet, self).__init__()

        in_dim = net_params['in_dim']
        K = net_params['K']
        nb_chev_filter = net_params['nb_chev_filter']
        nb_time_filter = net_params['nb_time_filter']
        time_strides = net_params['time_strides']
        seq_out_len = net_params['seq_out_len']
        seq_in_len = net_params['seq_in_len']
        num_nodes = net_params['num_nodes']
        nb_block = net_params['nb_block']

        self.BlockList = nn.ModuleList([ASTGCN_block(DEVICE, in_dim, K, nb_chev_filter, nb_time_filter, time_strides, supports, num_nodes, seq_in_len)])
        self.BlockList.extend([ASTGCN_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, supports, num_nodes, seq_in_len//time_strides) for _ in range(nb_block-1)])

        self.final_conv = nn.Conv2d(int(seq_in_len/time_strides), seq_out_len, kernel_size=(1, nb_time_filter))
        self.init_pars()

    def init_pars(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x, dummy=None):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        x = x.transpose(2,1)
        for block in self.BlockList:
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        return output

