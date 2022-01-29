import torch
import time
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from GNN.model import GCN, SGC
from utils import *
from args import *

args = get_args()
set_seed(args.seed, args.cuda)

config = {
    'lr': tune.loguniform(1e-4, 1e-1),
    'weight_decay': tune.loguniform(1e-10, 1e-4)
}