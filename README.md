# deep_learning
TJU deep_learning course project

## 实验环境

```
python3.9
Torch = 1.10.0
```

需要的依赖

```
numpy
scipy
hyperopt
torch_geometric
tensorboard
```

使用pytorch-geometric加载数据集，pyg安装方法：

```shell
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

其中${TORCH}和${CUDA}分别对应相应软件的版本，通过以下命令查看版本

```shell
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```



## 数据集下载

| 数据集   | 来源                                                         | 图   | 节点  | 边    | 特征 | 标签 |
| -------- | ------------------------------------------------------------ | ---- | ----- | ----- | ---- | ---- |
| Cora     | “Collective classification in network data,” AI magazine,2008 | 1    | 2708  | 5429  | 1433 | 7    |
| Citeseer | “Collective classification in network data,” AI magazine,2008 | 1    | 3327  | 4732  | 3703 | 6    |
| Pubmed   | “Collective classification in network data,” AI magazine,2008 | 1    | 19717 | 44338 | 500  | 3    |

以上数据集均可以使用pyg以统一形式加载，无需特别下载。

原始数据位于raw_data下，下载链接

[citeseer](https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz)

[cora](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)

[Pubmed](https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz)

## 运行方式

在GNN目录下直接运行train.py，按照默认参数设置训练

```shell
python train.py
```

详细的参数设置执行

```shell
python tran.py --help
```



## 实验结果

SGC

| 数据集   | loss   | Accuracy | Time(s) |
| -------- | ------ | -------- | ------- |
| Cora     | 0.7997 | 0.8130   | 0.7400  |
| Citeseer | 1.0555 | 0.7120   | 0.8151  |
| Pubmed   | 0.5718 | 0.7950   | 0.4131  |

GCN

| 数据集   | loss   | Accuracy | Time(s) |
| -------- | ------ | -------- | ------- |
| Cora     | 0.7871 | 0.8190   | 6.3408  |
| Citeseer | 1.0324 | 0.7100   | 18.3622 |
| Pubmed   | 0.5956 | 0.7810   | 24.6462 |



