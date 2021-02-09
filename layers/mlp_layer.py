import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, inputs, hiddens, act, out_act):
        """
        Just an MLP.
        Parameters
        ----------
        inputs : int
            Input size.
        hiddens : list[int]
            List of hidden units of each dense layer.
        act : callable
            Activation function.
        out_act : bool
            Whether to apply activation on the output of the last dense layer.
        """
        super().__init__()

        self.W = nn.ModuleList()
        self.act = act
        self.out_act = out_act
        for i in range(len(hiddens)):
            in_dims = inputs if i == 0 else hiddens[i - 1]
            out_dims = hiddens[i]
            self.W.append(nn.Linear(in_dims, out_dims))

    def forward(self, x):
        for i, W in enumerate(self.W):
            x = W(x)
            if i != len(self.W) - 1 or self.out_act:
                x = self.act(x)
        return x
