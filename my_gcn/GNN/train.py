import torch
import time
from GNN.model import GCN, SGC
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import *
from args import *

args = get_args()
# writer = SummaryWriter()
set_seed(args.seed, args.cuda)
adj, features, labels, train_mask, val_mask, test_mask = load_data(args.dataset, args.cuda)

if args.model == 'SGC':
    features = sgc_precompute(features, adj, 2)
    model = SGC(features.shape[1], labels.max().item() + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=1.3e-5)
else:
    model = GCN(features.shape[1], 16, labels.max().item() + 1, 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

if args.cuda:
    model.cuda()


def train_gcn(epoch):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    out = model(features, adj)

    loss_train = F.nll_loss(out[train_mask], labels[train_mask])
    acc_train = accuracy(out[train_mask], labels[train_mask])

    # writer.add_scalar("Loss/train", loss_train, epoch)
    # writer.add_scalar("Accuracy/train", acc_train, epoch)

    loss_train.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        out = model(features, adj)
        loss_val = F.nll_loss(out[val_mask], labels[val_mask])
        acc_val = accuracy(out[val_mask], labels[val_mask])

    # writer.add_scalar("Loss/validation", loss_train, epoch)
    # writer.add_scalar("Accuracy/validation", acc_train, epoch)

    print(f'Epoch: {epoch+1:04d}',
          f'loss_train: {loss_train.item():.4f}',
          f'acc_train: {acc_train.item():.4f}',
          f'loss_val: {loss_val.item():.4f}',
          f'acc_val: {acc_val.item():.4f}',
          f'time: {time.time() - t0:.4f}')


def test_gcn():
    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = accuracy(output[test_mask], labels[test_mask])

    print("Test set results:",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test.item():.4f}")


def train_sgc(epoch):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features[train_mask])
    loss_train = F.cross_entropy(output, labels[train_mask])
    acc_train = accuracy(output, labels[train_mask])
    loss_train.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        output = model(features[val_mask])
        loss_val = F.cross_entropy(output, labels[val_mask])
        acc_val = accuracy(output, labels[val_mask])

    print(f'Epoch: {epoch+1:04d}',
          f'loss_train: {loss_train.item():.4f}',
          f'acc_train: {acc_train.item():.4f}',
          f'loss_val: {loss_val.item():.4f}',
          f'acc_val: {acc_val.item():.4f}',
          f'time: {time.time() - t0:.4f}')


def test_sgc():
    with torch.no_grad:
        model.eval()
        output = model(features[test_mask])
        loss_test = F.cross_entropy(output, labels[test_mask])
        acc_test = accuracy(output, labels[test_mask])

    print("Test set results:",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test.item():.4f}")


# Train model
t_total = time.time()
for epoch in range(200):
    train_gcn(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# writer.flush()

# Testing
test_gcn()
