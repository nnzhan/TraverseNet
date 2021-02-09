from torch import nn
from module.stgcn_block import STGCNBlock,TimeBlock

class STGCNnet(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, net_params, adj):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCNnet, self).__init__()

        num_nodes = net_params['num_nodes']
        num_features = net_params['in_dim']
        num_timesteps_input = net_params['seq_in_len']
        num_timesteps_output = net_params['seq_out_len']

        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=32, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=128,
                                 spatial_channels=32, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=128, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)
        self.adj = adj
    def forward(self, X, dummy):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, self.adj)
        out2 = self.block2(out1, self.adj)
        out3 = self.last_temporal(out2)
        out3 = out3.transpose(2,1)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        out4 = out4.unsqueeze(dim=1)
        out4 = out4.transpose(3,1)
        return out4

