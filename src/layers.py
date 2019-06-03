import math

import numpy as np
import torch
import torch.nn as nn

class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        output_dim : int
            Dimension of output node features.
        """
        super(GraphConvolution, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim)

        a = 1. / math.sqrt(output_dim)
        self.fc.weight.data.uniform_(-a, a)
        self.fc.bias.data.uniform_(-a, a)

    def forward(self, features, adj):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n x input_dim) tensor of input node features.
        adj : torch.sparse.LongTensor
            An (n x n) sparse tensor representing the normalized adjacency
          matrix of the graph.

        Returns
        -------
        out : torch.Tensor
            An (n x output_dim) tensor of output node features.
        """
        out = torch.sparse.mm(adj, features)
        out = self.fc(out)

        return out