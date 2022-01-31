import pickle as pkl
import os

# a = {1: '6666'}
# os.makedirs("./GCN-tuning", exist_ok=True)
# path = "./GCN-tuning/test.lx"
# with open(path, 'wb') as f:
#     pkl.dump(a, f)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.stochastic import sample
from math import log
space = [hp.loguniform('test', log(1e-3), log(1e-1))]
print(sample(space))
