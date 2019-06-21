import os

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

class Cora(Dataset):

    def __init__(self, path, num_layers, self_loop=False,
                 normalize_adj=False):
        """
        Parameters
        ----------
        path : str
            Path to the cora dataset with cora.cites and cora.content files.
        num_layers : int
            Depth of the model.
        self_loop : Boolean
            Whether to add self loops, default: False.
        normalize_adj : Boolean
            Whether to use symmetric normalization on the adjacency matrix, default: False.
        """
        super(Cora, self).__init__()

        self.path = path
        self.num_layers = num_layers
        self.self_loop = self_loop
        self.normalize_adj = normalize_adj
        self.idx = {
            'train' : np.array(range(140)),
            'val' : np.array(range(200, 500)),
            'test' : np.array(range(500, 1500))
        }

        print('--------------------------------')
        print('Reading cora dataset from {}'.format(path))
        citations = np.loadtxt(os.path.join(path, 'cora.cites'), dtype=np.int64)
        content = np.loadtxt(os.path.join(path, 'cora.content'), dtype=str)
        print('Finished reading data.')

        print('Setting up data structures.')
        features, labels = content[:, 1:-1].astype(np.float32), content[:, -1]
        d = {j : i for (i,j) in enumerate(sorted(set(labels)))}
        labels = np.array([d[l] for l in labels])

        vertices = np.array(content[:, 0], dtype=np.int64)
        d = {j : i for (i,j) in enumerate(vertices)}
        edges = np.array([e for e in citations if e[0] in d.keys() and e[1] in d.keys()])
        edges = np.array([d[v] for v in edges.flatten()]).reshape(edges.shape)
        n, m = labels.shape[0], edges.shape[0]
        u, v = edges[:, 0], edges[:, 1]
        adj = sp.coo_matrix((np.ones(m), (u, v)),
                            shape=(n, n),
                            dtype=np.float32)
        adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        if self_loop:
            adj += sp.eye(n)
        if normalize_adj:
            degrees = np.power(np.array(np.sum(adj, axis=1)), -0.5).flatten()
            degrees = sp.diags(degrees)
            adj = (degrees.dot(adj.dot(degrees)))
        print('Finished setting up data structures.')
        print('--------------------------------')

        self.features = features
        self.labels = labels
        self.adj = adj.tocoo()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.adj, self.features[idx], self.labels[idx]

    def get_dims(self):
        """
        Returns
        -------
        dimension of input features, dimension of output features
        """
        return self.features.shape[1], len(set(self.labels))

    def get_data(self):
        """
        Returns
        -------
        adj : torch.Sparse.LongTensor
            An (n x n) sparse tensor representing the normalized adjacency
            matrix of the graph.
        features : torch.Tensor
            An (n x input_dim) tensor of input node features.
        labels : torch.LongTensor
            An (n) length tensor of node labels.
        self.idx['train'] : numpy array
            Indices of the training nodes.
        self.idx['val'] : numpy array
            Indices of the validation nodes.
        self.idx['test'] : numpy array
            Indices of the testing nodes.
        """
        adj, features, labels = self._to_torch(self.adj, self.features,
                                               self.labels)
        return adj, features, labels, self.idx['train'], self.idx['val'], self.idx['test']

    def _to_torch(self, adj, features, labels):
        """
        Parameters
        ----------
        adj : scipy.sparse.coo_matrix() 
            An (n x n) sparse matrix representing the normalized adjacency
            matrix of the graph.
        features : numpy array
            An (n x input_dim) array of input node features.
        labels : numpy array
            An (n) length array of node labels.

        Returns
        -------
        adj : torch.sparse.LongTensor
            An (n x n) sparse tensor representing the normalized adjacency
            matrix of the graph.
        features : torch.Tensor
            An (n x input_dim) tensor of input node features.
        labels : torch.LongTensor
            An (n) length tensor of node labels.
        """
        indices = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
        values = torch.from_numpy(adj.data.astype(np.float32))
        shape = torch.Size(adj.shape)
        adj = torch.sparse.FloatTensor(indices, values, shape)
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        return adj, features, labels
