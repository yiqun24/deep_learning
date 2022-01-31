import torch
import time
import pickle as pkl
from model import GCN, SGC
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import *
from args import *

# 如果同时安装了tensorflow造成错误，执行以下代码
import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def train_gcn(features, labels, adj,
              train_mask, val_mask,
              hidden, dropout,
              lr, weight_decay,
              epochs, cuda):
    model = GCN(features.shape[1], hidden, labels.max().item() + 1, dropout)
    if cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    step = 0
    t_total = time.time()
    for epoch in range(epochs):
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

        # if epoch % 50 == 0:
        #     writer.add_embedding(out[train_mask], labels[train_mask], global_step=step)
        #     step += 1

        with torch.no_grad():
            model.eval()
            out = model(features, adj)
            loss_val = F.nll_loss(out[val_mask], labels[val_mask])
            acc_val = accuracy(out[val_mask], labels[val_mask])

        # writer.add_scalar("Loss/validation", loss_val, epoch)
        # writer.add_scalar("Accuracy/validation", acc_val, epoch)

        print(f'Epoch: {epoch + 1:04d}',
              f'loss_train: {loss_train.item():.4f}',
              f'acc_train: {acc_train.item():.4f}',
              f'loss_val: {loss_val.item():.4f}',
              f'acc_val: {acc_val.item():.4f}',
              f'time: {time.time() - t0:.4f}')

    print("Optimization Finished!")
    print(f"Total time elapsed: {time.time() - t_total:.4f}s")

    return model, (loss_val, acc_val)


def test_gcn(model, features, labels, adj, test_mask):
    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = accuracy(output[test_mask], labels[test_mask])

    # writer.add_embedding(output[test_mask], labels[test_mask], global_step=4)
    print("Test set results:",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test.item():.4f}")


def train_sgc(model,
              train_features, train_labels,
              val_features, val_labels,
              lr, weight_decay, epochs):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    t_total = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        acc_train = accuracy(output, train_labels)

        # writer.add_scalar("Loss/train", loss_train, epoch)
        # writer.add_scalar("Accuracy/train", acc_train, epoch)

        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            output = model(val_features)
            loss_val = F.cross_entropy(output, val_labels)
            acc_val = accuracy(output, val_labels)

        # writer.add_scalar("Loss/validation", loss_val, epoch)
        # writer.add_scalar("Accuracy/validation", acc_val, epoch)

        print(f'Epoch: {epoch + 1:04d}',
              f'loss_train: {loss_train.item():.4f}',
              f'acc_train: {acc_train.item():.4f}',
              f'loss_val: {loss_val.item():.4f}',
              f'acc_val: {acc_val.item():.4f}',
              f'time: {time.time() - t0:.4f}')

    print("Optimization Finished!")
    print(f"Total time elapsed: {time.time() - t_total:.4f}s")

    return model, (loss_val, acc_val)


def test_sgc(model, test_features, test_labels):
    with torch.no_grad():
        model.eval()
        output = model(test_features)
        loss_test = F.cross_entropy(output, test_labels)
        acc_test = accuracy(output, test_labels)

    print("Test set results:",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test.item():.4f}")


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed, args.cuda)

    # 用于追踪训练过程从而使用tensorboard进行可视化
    # writer = SummaryWriter()

    if args.tuned:
        with open(f"../{args.model}-tuning/{args.dataset}.tune", 'rb') as f:
            paras = pkl.load(f)
            args.lr = round(paras['lr'], 2)
            args.weight_decay = paras['weight_decay']
            print(f"using tuned weight decay: {args.weight_decay} " +
                  f"using tuned learning rate: {args.lr}")

    adj, features, labels, train_mask, val_mask, test_mask = load_data(args.dataset, args.cuda)

    if args.model == 'GCN':
        trained_model, _ = train_gcn(features, labels, adj, train_mask, val_mask,
                                     args.hidden, args.dropout, args.lr,
                                     args.weight_decay, args.epochs, args.cuda)
        test_gcn(trained_model, features, labels, adj, test_mask)

    elif args.model == 'SGC':
        features = sgc_precompute(features, adj, args.degree)
        SGC_model = SGC(features.shape[1], labels.max().item() + 1)
        trained_model, _ = train_sgc(SGC_model, features[train_mask], labels[train_mask],
                                     features[val_mask], labels[val_mask],
                                     args.lr, args.weight_decay, args.epochs)
        test_sgc(trained_model, features[test_mask], labels[test_mask])

    # writer.flush()
