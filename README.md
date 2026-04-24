# Homework 1: NumPy MLP on EuroSAT RGB

本仓库是《深度学习与空间智能》`Homework 1` 的本地实现与实验结果整理版本。项目使用纯 `NumPy` 实现三层参数层的 MLP（输入层 -> 两层隐藏层 -> 输出层），并在 `EuroSAT RGB` 数据集上完成训练、验证、测试、可视化与误差分析。

## 项目内容

- 纯 `NumPy` 实现前向传播、反向传播、参数更新。
- 使用 `SGD`、`Cross-Entropy Loss`、`Learning Rate Decay` 和 `L2 (Weight Decay)` 训练模型。
- 使用验证集进行超参数搜索和最优模型选择。
- 生成训练集/验证集 `Loss` 曲线、验证集 `Accuracy` 曲线、第一层隐藏层权重可视化、混淆矩阵和错分样例图。
- 完整保存 `240` 组网格搜索结果，并单独整理最优模型提交材料。

## 报告与提交链接

实验报告：
- `report.pdf`
- GitHub Repo：`https://github.com/MayRainStop/eurosat-mlp-classification`
- 模型权重下载地址：`https://drive.google.com/file/d/1eAFBxHwdhvLdK9Uq4B-5a0SveSodhju4/view?usp=sharing`

## 仓库目录结构

PS: 目录里包含了放在Google Drive的原始数据、模型权重部分

```text
.
|- EuroSAT_RGB/
|- mlp_numpy/
|- results/
|  |- best_model/
|  |- relu/
|  |- tanh/
|  |- grid_summary.csv
|  `- grid_summary.json
|- train.py
|- evaluate.py
|- run_experiments.py
|- visualize_model.py
|- report.tex
|- report.pdf
|- README.md
`- hw1.pdf
```

各目录和文件说明如下：

- `EuroSAT_RGB/`：原始数据集目录，类别子文件夹保持题目给定结构不变。
- `mlp_numpy/`：核心实现代码，包括数据读取、MLP、训练器、评估指标与可视化函数。
- `results/best_model/`：最优模型及报告提交所需图表。
- `results/relu/`：按 `ReLU` 激活函数整理的网格搜索实验结果。
- `results/tanh/`：按 `Tanh` 激活函数整理的网格搜索实验结果。
- `results/grid_summary.csv`、`results/grid_summary.json`：全部实验汇总结果。
- `train.py`：训练单个模型配置。
- `evaluate.py`：加载保存好的模型并在测试集上评估。
- `run_experiments.py`：执行网格搜索。
- `visualize_model.py`：根据已有最优模型结果重新生成报告所需图像。
- `report.tex`、`report.pdf`：实验报告源码与编译后的 PDF。
- `hw1.pdf`：作业原始要求文件。

## 环境依赖

建议使用 `Python 3.10+`。主要依赖如下：

- `numpy`
- `matplotlib`
- `Pillow`

如果需要重新编译实验报告，还需要安装：

- `TeX Live`
- `XeLaTeX`

可以使用如下命令安装 Python 依赖：

```bash
pip install numpy matplotlib pillow
```

## 数据集放置方式

数据集应放在仓库根目录下，目录结构如下：

```text
EuroSAT_RGB/
|- AnnualCrop/
|- Forest/
|- HerbaceousVegetation/
|- Highway/
|- Industrial/
|- Pasture/
|- PermanentCrop/
|- Residential/
|- River/
`- SeaLake/
```

数据读取时会自动跳过一张损坏图片：

```text
EuroSAT_RGB/HerbaceousVegetation/HerbaceousVegetation_2468.jpg
```

## 如何训练最优模型

运行以下命令可训练当前最优配置：

```bash
python train.py \
  --data-dir EuroSAT_RGB \
  --output-dir results/best_model \
  --image-size 32 \
  --hidden-dims 2048 256 \
  --activation relu \
  --epochs 200 \
  --batch-size 128 \
  --learning-rate 0.1 \
  --lr-decay 0.95 \
  --weight-decay 0.001 \
  --patience 10
```

最优配置为：

```text
hidden_dims = (2048, 256)
activation = ReLU
learning_rate = 0.1
weight_decay = 1e-3
epochs = 200
patience = 10
```

## 如何测试最优模型

运行以下命令可评估保存好的最优模型：

```bash
python evaluate.py \
  --data-dir EuroSAT_RGB \
  --checkpoint results/best_model/best_model.npz \
  --output-dir results/evaluation \
  --image-size 32 \
  --hidden-dims 2048 256 \
  --activation relu \
  --weight-decay 0.001
```

## 如何重新生成报告图像

运行以下命令可基于已有最优模型结果重新生成报告中的可视化图像：

```bash
python visualize_model.py \
  --run-dir results/best_model \
  --data-dir EuroSAT_RGB \
  --image-size 32 \
  --weight-grid-columns 64 \
  --top-k-weights 64
```

## 如何运行超参数网格搜索

运行以下命令可执行完整网格搜索：

```bash
python run_experiments.py \
  --data-dir EuroSAT_RGB \
  --output-dir results/search_runs \
  --image-size 32 \
  --hidden-dim1s 2048 1024 512 \
  --hidden-dim2s 1024 512 256 128 \
  --activations relu tanh \
  --learning-rates 0.003 0.01 0.03 0.1 \
  --weight-decays 1e-5 1e-4 1e-3 \
  --epochs 200 \
  --patience 10
```

说明：

- 上述搜索空间共对应 `240` 组有效实验。
- `run_experiments.py` 会自动跳过 `hidden_dim2 > hidden_dim1` 的组合。
- `learning_rate = 0.3` 没有纳入正式网格搜索，因为额外探测实验显示该学习率会快速发散，出现 `overflow/NaN`，验证集准确率也退化到接近随机猜测水平。

## 最优模型结果

当前最佳结果如下：

```text
best_epoch = 58
stopped_epoch = 68
best_val_accuracy = 0.7152
test_accuracy = 0.7053
```

`results/best_model/` 中的主要文件说明如下：

- `best_model.npz`：最优模型参数。
- `history.json`：逐 epoch 的训练/验证损失与准确率。
- `metrics.json`：最优模型的详细指标、混淆矩阵和分类结果统计。
- `loss_curves.png`：训练集和验证集 `Loss` 曲线。
- `validation_accuracy_curve.png`：验证集 `Accuracy` 曲线。
- `first_layer_weights.png`：完整第一层隐藏层权重可视化。
- `first_layer_weights_top64.png`：较适合放入正文展示的第一层权重子集图。
- `confusion_matrix.png`：测试集混淆矩阵。
- `misclassified_examples.png`：测试集错分样例。

