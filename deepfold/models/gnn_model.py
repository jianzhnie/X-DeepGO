import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import (ChebConv, GCNConv, GINConv, GMMConv,
                                global_add_pool)


class Embedder(nn.Module):
    def __init__(self, go_size, hidden_dimension):
        super().__init__()
        self.embed = nn.Linear(go_size, hidden_dimension)

    def forward(self, x):
        node_feature = self.embed(x)
        node_feature = F.normalize(node_feature)
        return node_feature


class GraphConvolution(nn.Module):
    def __init__(self, nfeat, nhid, bias=True):
        super(GraphConvolution, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.weight = Parameter(torch.FloatTensor(nfeat, nhid))
        if bias:
            self.bias = Parameter(torch.FloatTensor(nhid))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = torch.mm(input, self.weight)
        output = torch.spmm(adj, x)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class CustomGCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, output_dim, bias=True),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        return self.mlp(x)


class GraphCNN(nn.Module):
    def __init__(self,
                 input_dim=1024,
                 net_type='gcn',
                 channel_dims=[256, 256, 512],
                 fc_dim=512,
                 num_classes=256,
                 cheb_order=2):
        super(GraphCNN, self).__init__()

        # Define graph convolutional layers
        gcn_dims = [input_dim] + channel_dims
        self.net_type = net_type
        if net_type == 'gcn':
            gcn_layers = [
                GCNConv(gcn_dims[i - 1], gcn_dims[i], bias=True)
                for i in range(1, len(gcn_dims))
            ]
        elif net_type == 'chebcn':
            gcn_layers = [
                ChebConv(gcn_dims[i - 1],
                         gcn_dims[i],
                         K=cheb_order,
                         normalization='sym',
                         bias=True) for i in range(1, len(gcn_dims))
            ]
        elif net_type == 'gmmcn':
            gcn_layers = [
                GMMConv(gcn_dims[i - 1],
                        gcn_dims[i],
                        dim=1,
                        kernel_size=1,
                        separate_gaussians=False,
                        bias=True) for i in range(1, len(gcn_dims))
            ]
        elif net_type == 'gincn':
            gcn_layers = [
                GINConv(MLP(gcn_dims[i - 1], gcn_dims[i], gcn_dims[i]),
                        eps=0,
                        train_eps=False) for i in range(1, len(gcn_dims))
            ]
        self.gcn = nn.ModuleList(gcn_layers)

        # Define dropout
        self.drop = nn.Dropout(p=0.3)

        # Define fully-connected layers
        self.fc_dim = fc_dim
        if self.fc_dim > 0:
            self.fc = nn.Linear(channel_dims[-1], fc_dim)
            self.fc_out = nn.Linear(fc_dim, num_classes)
        else:
            self.fc_out = nn.Linear(channel_dims[-1], num_classes)

    def forward(self, data):
        x = data.x

        # Compute graph convolutional part
        if self.net_type != 'gmmcn':
            for gcn_layer in self.gcn:
                x = F.relu(gcn_layer(x, data.edge_index))
        else:
            for gcn_layer in self.gcn:
                x = F.relu(
                    gcn_layer(x.float(), data.edge_index.long(),
                              data.pseudo.float()))

        # Apply global sum pooling and dropout
        x = global_add_pool(x, data.batch)
        x = self.drop(x)
        embedding = x

        # Compute fully-connected part
        if self.fc_dim > 0:
            x = F.relu(self.fc(x))

        output = self.fc_out(x)  # sigmoid in loss function

        return embedding, output
