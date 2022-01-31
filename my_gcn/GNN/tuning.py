import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import pickle as pkl
from utils import *
from args import *
from train import train_gcn
from math import log
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

args = get_args()
set_seed(args.seed, args.cuda)

adj, features, labels, train_mask, val_mask, test_mask = load_data(args.dataset, args.cuda)

space = {'weight_decay': hp.loguniform('weight_decay', log(1e-4), log(1e-3)),
         'lr': hp.loguniform('lr', log(1e-3), log(1e-1))}


def gcn_objective(space):
    trained_model, (loss_val, acc_val) = train_gcn(features, labels, adj, train_mask, val_mask,
                                                   args.hidden, args.dropout, space['lr'],
                                                   space['weight_decay'], args.epochs, args.cuda)
    print(f"learning rate: {space['lr']:.4e} " + f"weight decay: {space['weight_decay']:.3e} " +
          f"accuracy: {acc_val:.4f}")
    return {'loss': -acc_val, 'status': STATUS_OK}


best = fmin(gcn_objective, space=space, algo=tpe.suggest, max_evals=100)
print(best)
print(f"Best learning rate: {best['lr']:.4e} " +
      f"Best weight decay: {best['weight_decay']:.4e} ")

os.makedirs(f"../{args.model}-tuning", exist_ok=True)
path = f'../{args.model}-tuning/{args.dataset}.tune'
with open(path, 'wb') as f:
    pkl.dump(best, f)
