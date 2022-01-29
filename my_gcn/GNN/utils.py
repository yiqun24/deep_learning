import numpy as np
import scipy.sparse as sp
import torch.nn.functional
from torch_geometric.datasets import Planetoid


def load_data(d_name, cuda):
    dataset = Planetoid(root=f'../dataset/{d_name}', name=d_name)
    data = dataset[0]

    labels = data.y
    features = sp.csr_matrix(data.x, dtype=np.float32)
    edges = data.edge_index.T.contiguous()
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    features = row_normalize(features)
    adj = adj_normalize(adj + sp.eye(adj.shape[0]))

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    features = torch.Tensor(features.toarray())
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    return adj, features, labels, train_mask, val_mask, test_mask


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def adj_normalize(adj):
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return adj


def row_normalize(mx):
    row_sum = np.array(mx.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sgc_precompute(features, adj, degree):
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features


def accuracy(output, labels):
    predictions = output.max(1)[1].type_as(labels)
    correct = predictions.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
