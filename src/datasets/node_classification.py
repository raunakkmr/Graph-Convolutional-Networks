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

class ContactHS(Dataset):

    def __init__(self, path, mode, num_layers, self_loop=False, normalize_adj=False):
        super().__init__()

        self.path = path
        self.mode = mode
        self.num_layers = num_layers
        self.self_loop = self_loop
        self.normalize_adj = normalize_adj
        self.idx = {
            'train' : np.array(range(65)),
            'val' : np.array(range(65, 320)),
            'test' : np.array(range(65, 320))
        }

        print('--------------------------------')
        print('Reading contact-high-school dataset from {}'.format(path))
        with open(os.path.join(path, 'contacts-HS-nodes.tsv'), 'r') as f:
            nodelines = f.readlines()[1:]
        with open(os.path.join(path, 'contacts-HS-hyperedges.tsv'), 'r') as f:
            hyperedgelines = f.readlines()[1:]
        print('Finished reading data.')

        print('Setting up data structures.')
        n = len(nodelines)
        idx = np.arange(n)
        features = np.eye(n)
        labels = [int(nodelines[i].split('\t')[1]) - 1 for i in idx]
        d = {j : i for (i,j) in enumerate(sorted(set(labels)))}
        # if mode == 'train':
        #     idx = self.idx['train']
        # elif mode == 'val':
        #     idx = np.hstack((self.idx['train'], self.idx['val']))
        # elif mode == 'test':
        #     idx = np.hstack((self.idx['train'], self.idx['test']))
        # labels = np.array([d[labels[i]] for i in idx])
        labels = np.array([d[l] for l in labels])

        vertices = [int(line.split('\t')[0]) - 1 for line in nodelines]
        vertices = np.array(vertices, dtype=np.int64)
        d = {j : i for (i,j) in enumerate(vertices)}
        hyperedges = [line.split('\t')[0] for line in hyperedgelines]
        edge_list = []
        for hyperedge in hyperedges:
            hyperedge_ = [int(x) for x in hyperedge.strip().split(',')]
            for i in range(len(hyperedge)):
                for j in range(i+1, len(hyperedge)):
                    edge_list.append([i, j])

        edges = np.unique(np.array([e for e in edge_list if e[0] in d.keys() and e[1] in d.keys()]), axis=0)
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
        # self.adj = adj.tolil()
        self.adj = adj.tocoo()

    def __len__(self):
        return len(self.idx[self.mode])

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

    def __getitem__(self, idx):
        return self.adj, self.features[idx], self.labels[idx]

    def collate_wrapper(self, batch):
        """
        Parameters
        ----------
        batch : list
            A list of examples from this dataset.

        Returns
        -------
        features : torch.FloatTensor
            An (n' x input_dim) tensor of input node features.
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        rows : list
        labels : torch.LongTensor
            An (n') length tensor of node labels.
        """
        idx = [node_layers[-1][0] for node_layers in [sample[1] for sample in batch]]

        node_layers, mappings = self._form_computation_graph(idx)
        rows = self.adj.rows[node_layers[0]]
        features = self.features[node_layers[0], :]
        labels = self.labels[node_layers[-1]]
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        return features, node_layers, mappings, rows, labels

    def get_dims(self):
        """
        Returns
        -------
        dimension of input features, dimension of output features
        """
        return self.features.shape[1], len(set(self.labels))

    def _form_computation_graph(self, idx):
        """
        Parameters
        ----------
        idx : int
            Index of the node for which the forward pass needs to be computed.

        Returns
        -------
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        """
        _list, _set = list, set
        rows = self.adj.rows
        if type(idx) is int:
            node_layers = [np.array([idx], dtype=np.int64)]
        elif type(idx) is list:
            node_layers = [np.array(idx, dtype=np.int64)]
        for _ in range(self.num_layers):
            prev = node_layers[-1]
            arr = [node for node in prev]
            arr.extend([v for node in arr for v in rows[node]])
            arr = np.array(_list(_set(arr)), dtype=np.int64)
            node_layers.append(arr)
        node_layers.reverse()

        mappings = [{j : i for (i,j) in enumerate(arr)} for arr in node_layers]

        return node_layers, mappings
