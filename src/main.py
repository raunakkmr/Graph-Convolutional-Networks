import argparse
import json
from math import ceil
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import datasets
import models
import utils

def main():
    config = utils.parse_args()

    if config['cuda'] and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    dataset_args = (config['task'], config['dataset'], config['dataset_path'],
                    config['num_layers'], config['self_loop'],
                    config['normalize_adj'])
    dataset = utils.get_dataset(dataset_args)

    input_dim, output_dim = dataset.get_dims()
    adj, features, labels, idx_train, idx_val, idx_test = dataset.get_data()
    x = features
    y_train = labels[idx_train]
    y_val = labels[idx_val]
    y_test = labels[idx_test]

    model = models.GCN(input_dim, config['hidden_dims'], output_dim,
                       config['dropout'])
    model.to(device)

    if not config['load']:
        criterion = utils.get_criterion(config['task'])
        optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                            weight_decay=config['weight_decay'])
        epochs = config['epochs']
        model.train()
        print('--------------------------------')
        print('Training.')
        for epoch in range(epochs):
            optimizer.zero_grad()
            scores = model(x, adj)[idx_train]
            loss = criterion(scores, y_train)
            loss.backward()
            optimizer.step()
            predictions = torch.max(scores, dim=1)[1]
            num_correct = torch.sum(predictions == y_train).item()
            accuracy = num_correct / len(y_train)
            print('    Training epoch: {}, loss: {:.3f}, accuracy: {:.2f}'.format(
                epoch+1, loss.item(), accuracy))
        print('Finished training.')
        print('--------------------------------')

        if config['save']:
            print('--------------------------------')
            directory = os.path.join(os.path.dirname(os.getcwd()),
                                    'trained_models')
            if not os.path.exists(directory):
                os.makedirs(directory)
            fname = utils.get_fname(config)
            path = os.path.join(directory, fname)
            print('Saving model at {}'.format(path))
            torch.save(model.state_dict(), path)
            print('Finished saving model.')
            print('--------------------------------')

    if config['load']:
        directory = os.path.join(os.path.dirname(os.getcwd()),
                                 'trained_models')
        fname = utils.get_fname(config)
        path = os.path.join(directory, fname)
        model.load_state_dict(torch.load(path))
    model.eval()
    print('--------------------------------')
    print('Testing.')
    scores = model(x, adj)[idx_test]
    predictions = torch.max(scores, dim=1)[1]
    num_correct = torch.sum(predictions == y_test).item()
    accuracy = num_correct / len(y_test)
    print('    Test accuracy: {}'.format(accuracy))
    print('Finished testing.')
    print('--------------------------------')

if __name__ == '__main__':
    main()