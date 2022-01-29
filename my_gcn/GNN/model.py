import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        stdv = 1. / np.sqrt(out_features)
        self.weight = Parameter(torch.Tensor(in_features, out_features).uniform_(-stdv, stdv))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).uniform_(-stdv, stdv))
        else:
            self.bias = None

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output


class GCN(nn.Module):
    def __init__(self, n_feats, n_hid, n_classes, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_feats, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_classes)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.dropout1(x)
        x = F.relu(self.gc1(x, adj))
        x = self.dropout2(x)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class SGC(nn.Module):
    def __init__(self, n_feats, n_classes):
        super(SGC, self).__init__()
        self.W = nn.Linear(n_feats, n_classes)

    def forward(self, x):
        return self.W(x)

