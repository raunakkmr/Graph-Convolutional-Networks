import numpy as np
import torch
import torch.nn as nn

import layers

class GCN(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim,
                 dropout=0.5):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        hidden_dims : list of ints
            Dimensions of hidden layers. Must be non empty.
        output_dim : int
            Dimension of output node features.
        dropout : float
            Dropout rate. Default: 0.5.
        """
        super(GCN, self).__init__()

        self.convs = nn.ModuleList([layers.GraphConvolution(input_dim, hidden_dims[0])])
        self.convs.extend([layers.GraphConvolution(hidden_dims[i-1], hidden_dims[i]) for i in range(1, len(hidden_dims))])
        self.convs.extend([layers.GraphConvolution(hidden_dims[-1], output_dim)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

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
        out = features
        for conv in self.convs[:-1]:
            out = self.dropout(self.relu(conv(out, adj)))
        out = self.convs[-1](out, adj)

        return out