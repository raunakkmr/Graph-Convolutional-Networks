# Graph-Convolutional-Networks
This is a PyTorch implementation of graph convolutional networks (GCNs) from
the paper [Semi-Supervised Classification with Graph Convolutional
Networks](https://arxiv.org/abs/1609.02907).

## Usage

In the `src` directory, edit the `config.json` file to specify arguments and
flags. Then run `python main.py`.

## Limitations
* Does not support mini-batch training.
* Currently, only supports the Cora dataset. However, for a new dataset it should be fairly straightforward to write a Dataset class similar to `datasets.Cora`.

## References
* [Semi-Supervised Classification with Graph Convolutional
Networks](https://arxiv.org/abs/1609.02907), Kipf and Welling, ICLR 2017.
* [Collective Classification in Network Data](https://www.aaai.org/ojs/index.php/aimagazine/article/view/2157), Sen et al., AI Magazine 2008.