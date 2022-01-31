import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=10, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default="Cora",
                        help='Dataset to use.(Cora, Citeseer, Pubmed)')
    parser.add_argument('--model', type=str, default="GCN",
                        choices=["SGC", "GCN"],
                        help='model to use.')
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')

    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
