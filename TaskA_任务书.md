# AIR5011 / MDS5122 — Assignment 1 · Task A 任务书

> **性质说明**：这是一份"脚手架"，不是答案。全文只告诉你**做什么、为什么、查什么、怎么自检**，
> 代码由你自己写。凡是出现"✍️"标记的地方，都是留给你动手的。
>
> **提交要求回顾**：截止 2026-10-21 23:59（Blackboard）。PDF 报告 + 代码 zip **分开上传**，
> 报告不要放进 zip。报告用英文写。

---

## 0. 先读这里：任务全景

### 0.1 Task A 到底在考什么

任务书原文有一句最关键的话：

> "we do not score your assignment solely based on the achieved accuracy;
> instead, we seek evidence of your effort and multiple attempts to improve accuracy incrementally"

翻译成人话：**这不是一次性能竞赛，是一次"实验方法论"的展示**。
40 分里只有 10 分和"跑通 baseline"有关，20 分全在"你如何系统性地、可解释地提升精度"。

所以你的目标不是"调出一个高分模型"，而是**产出一条能被复现、被讲清楚的实验链**：

```
baseline ──[改 A，+3.2%]──> 版本1 ──[改 B，+5.1%]──> 版本2 ──[改 C，+2.4%]──> 最终模型
```

每一步都要能回答三个问题：**我改了什么？为什么觉得它会有效？实测涨了多少？**

### 0.2 评分拆解与工时预算

| 评分项 | 分值 | 对应阶段 | 建议投入 |
|---|---|---|---|
| 1. 环境 + 简单 CNN，10 epochs，报 baseline 精度 | 10 | 第 0–2 阶段 | 1–2 天 |
| 2. 提升精度的因素探索 | 20 | 第 3 阶段 ★ | 4–6 天（主战场） |
| 3. 代码与文档整洁、表格、高质量图表 | 5 | 第 4 阶段 | 1–2 天 |
| 4. 学到了什么 | 5 | 第 4 阶段 | 半天 |

### 0.3 题目给定的 baseline 架构（这是题目给的，不是你的答案）

原文用一张图给出，必须**原样实现**作为对照基准：

```python
model = nn.Sequential(
    nn.Conv2d(3, 256, 5, padding=2), nn.ReLU(),
    nn.Conv2d(256, 256, 5, padding=2), nn.ReLU(),
    nn.Conv2d(256, 256, 5, padding=2), nn.ReLU(),
    nn.Conv2d(256, 256, 5, padding=2), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(256, 100)
)
```

**先读懂这个架构，它决定了后面所有实验的起点。** 三个观察：

| 观察 | 含义 | 后果 |
|---|---|---|
| 4 层全是 5×5，`padding=2`，**没有池化/下采样** | 全程在 32×32 分辨率上算 | 计算量巨大；且没有空间信息压缩，最后一层直接 GAP |
| **没有 BatchNorm / Dropout** | 纯裸卷积堆叠 | 训练脆弱、对学习率极敏感、易过拟合 |
| 通道数从 3 一步跳到 256 | 极宽的 stem | 参数和显存开销集中在第一层附近 |

> 🔍 **自检点 1**：手动数一遍参数量，应该得到 **4,961,124**。
> 手算公式：`Conv2d(Cin,Cout,k)` 参数 = `Cin×Cout×k×k + Cout`（bias 存在时）。
> 如果你的代码打印出这个数，说明架构复刻正确。
>
> 🔍 **自检点 2**：用 `torchsummary` 或手动 hook，确认每个 conv 的**输出 feature map 尺寸都是 32×32**。
> 理解 `padding = (k-1)/2` 为何让尺寸不变（奇数核保持尺寸的通用公式）。
>
> 💡 **思考题**（写进报告能加分）：这个设计相比"3×3 堆叠 + 下采样"的主流设计，
> 在**同等参数量**下的**计算量**和**感受野**分别是什么关系？为什么会说它是个"故意朴素"的 baseline？

---

## 第 0 阶段：环境准备（一次性，约 30–60 分钟）

### ⚠️ 0.1 第一优先级：你现在的 PyTorch 是 CPU 版本，必须换成 CUDA 版本

我实测了你的环境，这是本次作业**最大的风险点**：

| 项目 | 现状 |
|---|---|
| 显卡 | **NVIDIA GeForce RTX 4060 Laptop GPU**（8 GB 显存），驱动 591.74 / CUDA 13.1 |
| 已装 torch | `2.14.1+cpu` ← **CPU 构建，完全没用上显卡** |
| 你的 venv | `C:\Users\sigridwicks\Desktop\DL HW1\.venv` |

**实测性能（CPU 构建，batch=128）**：

```
训练速度   9 img/s
单 epoch   91 分钟
baseline 10 epochs  ≈  15.2 小时
```

而这只是**一次** baseline。第 3 阶段你至少要跑 20–40 组对比实验，按这个速度需要**几个月**。

**修复方式**（版本号完全不变，只是换成 CUDA 构建，`torch 2.14.1+cu126` 已确认存在）：

```bash
# 必须先卸载 CPU 版，否则 pip 可能认为"已满足"而不替换
.venv/Scripts/python.exe -m pip uninstall -y torch torchvision torchaudio

# 装 CUDA 12.6 构建
.venv/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> 💡 `cu126` / `cu128` / `cu129` 怎么选？你的驱动支持到 CUDA 13.1，三个都能用。
> `cu126` 与你现有版本号完全对应，最稳妥。下载约 2–3 GB，国内可能较慢，
> 可考虑先用清华镜像装 CPU 版做功能开发，等正式训练前再换 CUDA 版。
> **但正式跑实验前一定要换掉**——这是 15 小时和 20 分钟的差别。

**装完立刻验证** 👇

```python
import torch
print(torch.__version__)          # 期望看到 2.14.1+cu126
print(torch.cuda.is_available())  # 必须是 True
print(torch.cuda.get_device_name(0))
```

再跑一次同样的 benchmark，记录新的 img/s。**把这个"CPU vs GPU 训练速度对比"写进报告**，
是很漂亮的工程细节，也解释了为什么后面能负担更多实验。

> ⚠️ **8 GB 显存的注意事项**
> - batch size 从 128 起步，逐步试到 256/512，用 `nvidia-smi` 盯着显存。
> - 开 AMP（`torch.autocast`）能省显存又能提速，见 3.3 节。
> - 不要同时开着游戏 / 浏览器大量标签页，显存会被吃掉。
> - 报错 `CUDA out of memory` 时就降 batch size，**不要**去调 `torch.cuda.empty_cache()` 当万能药。



> [!NOTE]  
>
> 已完成，但是torch版本是2.11.0+cu128



### 0.2 补齐缺失的依赖

你的 venv 目前只有 `torch / torchvision / numpy / torchaudio`。作业**明确要求 Jupyter Notebook**，
还需要画图库。缺的：

| 包 | 用途 |
|---|---|
| `jupyter` / `ipykernel` | 作业硬性要求（"must be written using Jupyter Notebooks"） |
| `matplotlib` | 所有曲线图、可视化 |
| `tqdm` | 训练进度条（长实验必备，不然你会不知道跑到哪了） |
| `pandas` | 汇总实验结果表（强烈推荐，见 3.1） |
| `torchinfo`（原 torch-summary） | 一键打印每层输出尺寸和参数量，替代手算 |

```bash
.venv/Scripts/python.exe -m pip install jupyter ipykernel matplotlib tqdm pandas torchinfo
```

> 💡 如果你更想用 VS Code 写 notebook：装 `ipykernel` 就够了，
> 然后在 VS Code 里选 `.venv` 作为 kernel。注意 kernel 一定要选中你**装了 CUDA torch 的那个**环境，
> 选错环境是"我明明装了 GPU 版怎么还是慢"的头号原因。

### 0.3 项目结构（一开始就分好，别事后重构）

```
DL HW1/
├── .venv/
├── Assignment_1.pdf
├── data/                    # CIFAR-100 会自动下载到这里
├── notebooks/
│   ├── 01_eda.ipynb             # 数据探索与可视化
│   ├── 02_baseline.ipynb        # 10 分：baseline 复刻 + 训练
│   ├── 03_experiments.ipynb     # 20 分：提升实验
│   └── 04_final_analysis.ipynb  # 最终模型分析、图表产出
├── src/                     # 可复用的代码模块（notebook 里 import 它）
│   ├── data.py
│   ├── models.py
│   ├── engine.py            # 训练/评估循环
│   └── utils.py             # seed、计时、日志
├── runs/                    # 每次实验的日志/曲线/权重
│   └── exp_001_baseline/
│       ├── config.json
│       ├── metrics.csv
│       ├── curves.png
│       └── best.pt
└── report/
    └── main.tex
```

> 💡 **为什么要把代码放进 `src/` 而不是全写在 notebook 里？**
> 1. 你在 20 个实验里要反复用同一个训练循环。写在 notebook 里 = 复制粘贴 20 遍 = 改一处漏十九处。
> 2. 评分项 3 明确要求代码"clean and well-organized"。
> 3. notebook 保留**叙事和图表**，`src/` 保留**逻辑**，这是 Lab 里被推崇的实践。
>
> 实操：notebook 里用 `import sys; sys.path.append('..')` 然后 `from src.models import ...`。
> 改了 `src/` 里的文件后记得重启 kernel，否则 import 的是缓存。

### 0.4 复现性：现在不做，事后追悔

在**每个** notebook / 脚本开头固定：

```python
import random, numpy as np, torch

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

> ⚠️ **要有正确预期**：即使固定 seed，GPU 上的 conv 因为 cudnn 的算法选择，
> 结果也**不可能 100% 位级复现**，通常有 ±0.1~0.3% 的抖动。
> 这有个重要推论 👇
>
> 🔍 **判断"涨点是否真实"的经验法则**：两次运行的 seed 抖动约 ±0.2–0.5%。
> 所以**提升小于 0.5% 的改动，不要轻易下结论**，要么多跑几个 seed 取均值，
> 要么在报告里诚实地说"差异在噪声范围内"。**这句话本身就是加分项**——
> 它证明你懂实验误差，而不是无脑报个数字。
>
> 💡 `cudnn.benchmark = True` 能提速（自动找最快的卷积算法），但会让结果不可复现。
> 建议：开发调试时 `True`，出最终数据时 `False`。

### ✅ 第 0 阶段验收

- [x] `torch.cuda.is_available() == True`
- [ ] `nvidia-smi` 在训练时能看到显存占用和 GPU-Util 上升
- [x] Jupyter/VS Code 能选中 `.venv` kernel 并 `import torch` 成功
- [x] 目录结构建好，`src/utils.py` 里有 `set_seed`
- [ ] 记录下 GPU 版的新 benchmark 速度（img/s）

---

## 第 1 阶段：数据管线（约半天）

### 1.1 下载与加载

查阅 `torchvision.datasets.CIFAR100` 的文档，搞清楚每个参数：

| 参数 | 你要想清楚的问题 |
|---|---|
| `root` | 指向你的 `data/` 目录 |
| `train=True/False` | 训练集 50,000 张，测试集 10,000 张 |
| `download=True` | 首次会自动下载（约 169 MB）到 `root` |
| `transform` | 见 1.3，**这是后面涨分的主要杠杆之一** |

✍️ 自己写：分别构造 train / test 两个 Dataset 对象，然后各取一个样本，
打印 `type(img), img.shape, img.dtype, type(label), label`。

> 🔍 **自检点**：`img` 是 PIL Image 还是 Tensor？channel 在前还是在后？
> 尺寸是 `32×32` 还是别的？这个"格式认知"直接决定 transform 怎么写。

### 1.2 计算 CIFAR-100 的均值与标准差（别抄 CIFAR-10 的）

**为什么要做归一化**：让每个输入通道的分布接近 `N(0,1)`。
这使损失曲面更"圆"，梯度下降的步长在每个方向上尺度一致，**对学习率更鲁棒**，
也让后面 BatchNorm 的输入更规范。不做归一化的模型通常收敛更慢、精度更低。

**为什么不能直接抄 CIFAR-10 的 (0.4914, 0.4822, 0.4465)**：CIFAR-100 的类别分布不同，
统计量也不一样。**应该自己在训练集上算**。

✍️ 自己写（这是很好的 numpy 练习）：

1. 建一个 `transform=ToTensor()`（**只做 ToTensor，绝不要加 Normalize 或任何增强**）的临时 Dataset。
2. 把全部 50,000 张图堆成一个大张量，形状 `(50000, 3, 32, 32)`。
3. 沿 `(0, 2, 3)` 三个轴求 mean 和 std，得到形状 `(3,)` 的两组值。
4. 打印出来。

> ⚠️ **两个必须注意的点**：
> - **只能在训练集上算统计量**，绝不能用测试集（数据泄漏！）。
> - 求 std 时如果先算 `mean`，注意要除以像素总数 `N×H×W`，不是 `N`。
>   即 `std = sqrt( ((x - mean)**2).sum() / (N*H*W) )`。维度搞错是这里最常见的 bug。
>
> 🔍 **交叉验证**：算出来的值应该和文献里广泛引用的 CIFAR-100 统计量很接近
> （你可以自己搜一下核对）。如果差很多，说明维度或归一化系数算错了。
>
> 💡 别用 50000 张全量算 `std` 时内存爆掉——CIFAR-100 全量约 600 MB（float32），一般没事，
> 但如果报了内存错，就用 `torch.mean/std` 分批累加。这个小插曲本身可以写进报告。

**把结果留下来**，后面所有实验都要用这组数：

```python
CIFAR100_MEAN = (...)   # ✍️ 填入你算出来的值
CIFAR100_STD  = (...)
```

### 1.3 设计 transform

**训练集和测试集的 transform 必须不同**，这是核心概念：

| 集合 | 应包含 | 绝不能包含 |
|---|---|---|
| 训练集 | `ToTensor` → `Normalize`，以及**数据增强**（随机变换） | 任何非确定性变换出现在验证时 |
| 测试集 | `ToTensor` → `Normalize`（**只有这两个**） | RandomCrop / Flip / Erasing 等一切随机操作 |

**为什么测试集不能增强**：你评估的是"模型对真实样本的确定性预测"。
测试时做随机变换会让指标随运气波动、且与部署时行为不一致。

✍️ 第 1 阶段先只写 `ToTensor + Normalize`，把增强留到第 3 阶段当实验变量。

### 1.4 DataLoader 与可视化

查阅 `torch.utils.data.DataLoader` 的参数：`batch_size`、`shuffle`、
`num_workers`、`pin_memory`、`drop_last`。

| 参数 | 建议 | 说明 |
|---|---|---|
| `shuffle` | 训练集 **True**，测试集 False | 训练必须打乱，否则每个 epoch 的 batch 组成相同，损害泛化 |
| `num_workers` | Windows 上**从 0 开始试** | 见下方陷阱 |
| `pin_memory` | 有 GPU 时 **True** | 加速 CPU→GPU 拷贝 |
| `drop_last` | 训练集可 True | 避免最后一个小 batch 让 BatchNorm 统计不稳（batch 太小时 BN 的方差估计很差） |

> ⚠️ **Windows 上的经典陷阱**：`num_workers > 0` 时，
> 如果代码没有放在 `if __name__ == '__main__':` 保护下，会报
> `RuntimeError: An attempt has been made to start a new process before...`
> 或直接卡死。**在 Jupyter 里通常设 `num_workers=0` 最稳**。
> 数据增强在 CPU 上是瓶颈时，再考虑开多进程并用 `persistent_workers=True`。

✍️ **可视化检查（这一步 90% 的人跳过，但它能救你一命）**：

画一个 4×8 的网格图，展示 32 张训练图 + 它们的类别标签。做两件事：

1. **反归一化后画图**。归一化后的张量直接 `imshow` 会是一片鬼片（有负数、颜色失真）。
   你要写一个把 `Normalize` 逆回去的函数（提示：`x * std + mean`，然后把维度从 `(C,H,W)` 调成 `(H,W,C)`）。
2. 肉眼确认图像正常、标签合理。

> 💡 **为什么值得做**：如果你把 `Normalize` 的均值和方差参数写反了、
> 或者把 `(C,H,W)` 和 `(H,W,C)` 搞混、或者 label 和 image 错位，
> **模型照样能训练、loss 照样会下降**，你只是白白损失几个点精度，而且永远查不出来。
> 可视化是唯一能提前抓住这类 bug 的手段。
>
> 📊 这张图直接可以放进报告的"数据探索"章节。

### ✅ 第 1 阶段验收

- [x] train/test Dataset 与 DataLoader 构建完成，能正常迭代
- [x] 自己计算出 CIFAR-100 的 mean/std，并解释为什么不能用测试集算
- [x] 训练集 transform ≠ 测试集 transform，且理由清楚
- [x] 能用反归一化函数画出正常彩色图像，标签与图对得上
- [ ] 报告里已有一个 `(batch, C, H, W)` 和 `(batch, label)` 形状的说明

---

## 第 2 阶段：Baseline 复刻（10 分，约半天）

### 2.1 严格复刻架构

把 0.3 节题目给的 `nn.Sequential` **一字不差**地写进 `src/models.py`，
包成一个返回 `nn.Module` 的函数，例如 `def baseline_cnn(num_classes=100): ...`。

> ⚠️ **这一阶段千万不要"顺手优化"**：
> 不要加 BatchNorm、不要换 3×3、不要加 dropout。
> baseline 的唯一价值是**作为参照系**。你提前优化了，后面所有"+x%"都失去意义。
>
> 评分项 1 要的是"跑通题目给的架构并报告精度"，改动它等于没做这题。

✍️ 写完用 `torchinfo.summary(model, input_size=(1,3,32,32))` 打一张表，
核对总参数量 **4,961,124**，并检查每层输出尺寸。

### 2.2 训练循环的解剖

这是整个作业**最重要的基本功**。五步顺序不能错：

```
1. optimizer.zero_grad()      # 清空上一轮累积的梯度
2. outputs = model(inputs)    # 前向
3. loss = criterion(outputs, labels)
4. loss.backward()            # 反向，梯度写进 .grad
5. optimizer.step()           # 用梯度更新参数
```

> 🔍 **想清楚这几个"为什么"**（报告里能体现理解深度）：
> - 为什么 `zero_grad` 必须在**前向之前**或反向之前？如果忘了会怎样？
>   （提示：PyTorch 默认**累积**梯度而不是覆盖——这是特性不是 bug，梯度累积就靠它）
> - 为什么 `loss.backward()` 之后、`step()` 之前不能提前 `zero_grad`？
> - 训练时 `model.train()`，评估时 `model.eval()` + `torch.no_grad()` 各自解决什么问题？
>   （提示：前者影响 Dropout/BatchNorm 的行为模式，后者影响是否构建计算图和显存占用）
> - 评估时**忘了 `model.eval()`** 会怎样？（在 baseline 上没有 BN/Dropout 所以看不出来，
>   但等你加了 BN，这会是致命的静默 bug）

✍️ 自己实现两个函数：

```
train_one_epoch(model, loader, criterion, optimizer, device) -> (avg_loss, accuracy)
evaluate(model, loader, criterion, device) -> (avg_loss, accuracy)
```

**关于 accuracy 的正确算法**（小细节，但很多人写错）：

```python
correct = (preds.argmax(dim=1) == labels).sum().item()
```

> ⚠️ 别用 `torch.max(preds, 1)[1]` 之后又 `.sum()` 得到标量再除——顺序无所谓，
> 但**别忘了 `.item()`** 把 tensor 转成 Python 数，否则你会在一整个 epoch 里
> 不断累积计算图，显存缓慢泄漏最后 OOM。

### 2.3 损失函数与优化器

| 组件 | 用什么 | 为什么 |
|---|---|---|
| Loss | `nn.CrossEntropyLoss()` | CIFAR-100 是单标签多分类 |
| Optimizer | **先用 `SGD(lr=1e-2)` 或 `Adam`** | baseline 要"朴素"，把调参空间留给第 3 阶段 |

> 💡 **一个必须知道的坑**：`nn.CrossEntropyLoss` 内部**已经包含了 `log_softmax`**。
> 所以你的模型**最后一层不能再加 Softmax**，否则等于做了两次，梯度被严重缩放，模型基本学不动。
> 这是初学者排名第一的静默 bug。
>
> 同理 `nn.BCEWithLogitsLoss` 已含 sigmoid。

### 2.4 跑 10 个 epochs

题目要求**恰好 10 epochs**。

✍️ 每个 epoch 记录：`train_loss, train_acc, test_loss, test_acc`，逐 epoch 打印。
强烈建议同时记 `epoch_time`。

训练时同时观察：

- **train_loss 是否在下降**？如果几乎不动 → 学习率太小 / 梯度断了 / loss 写错。
- **如果 loss 变成 `nan`** → 学习率太大 / 忘了归一化 / log(0)。
- **train_acc 和 test_acc 的差距** → baseline 无正则化，应该很快开始过拟合。

### 2.5 保存与归档

每次实验都要落盘，**这是你能不能写出第 3 阶段报告的前提**：

```
runs/exp_000_baseline/
├── config.json    # {"arch":"baseline","epochs":10,"lr":0.01,"batch":128,"aug":"none","seed":42}
├── metrics.csv    # epoch,train_loss,train_acc,test_acc,epoch_time
├── curves.png     # loss/acc 曲线
└── model.pt       # state_dict
```

> 💡 **强烈建议现在就写一个实验记录工具**（`src/utils.py` 里的 `Run` 类或几个函数）：
> 传入 config 字典 → 自动建目录 → 训练完自动存 metrics.csv 和 curves.png。
> 第 3 阶段你要跑几十次实验，**手抄数字到表格里一定会出错**，
> 而且最后画对比图时你会需要程序化地读取所有 `metrics.csv` 拼成一张总表。

### 2.6 报告 baseline

报告里要写清楚：
- 完整超参表（arch / epochs / batch / lr / optimizer / aug / seed / 设备）
- 10 个 epoch 的 loss 与 accuracy 曲线
- 最终 test accuracy = **你的 baseline 数字**

> 🔍 **合理性检查**：这个网络无 BN、无增强、只训 10 epoch，
> test accuracy 大致会落在**百分之二十几到三十几**的区间（取决于优化器）。
> 如果只有 ~1%（等于瞎猜），说明训练出了问题（大概率是 loss 配了 softmax，或 lr 不对）。
> 如果轻松超过 50%，那你大概率**不小心用了测试集训练**——赶紧查。
>
> 💡 也顺便报一下 **top-5 accuracy**。CIFAR-100 有 100 类，
> top-1 数字偏低时 top-5 能提供更有信息量的视角，报告里两个都给会更专业。

### ✅ 第 2 阶段验收

- [ ] 参数量 = 4,961,124（证明架构复刻正确）
- [ ] 训练循环五步顺序正确，能解释每一步为什么
- [ ] 恰好训练 10 epochs，记录了逐 epoch 指标
- [ ] 有 loss/acc 曲线图（训练集 + 测试集两条线画在一起）
- [ ] baseline test accuracy 已记录，且量级合理
- [ ] `runs/exp_000_baseline/` 里有 config、metrics、曲线、权重

---

## 第 3 阶段：提升精度（20 分）★ 主战场

### 3.1 先建方法论，再动手

**这是拿满 20 分和拿 10 分的分界线。** 在改任何一行代码之前，先把下面这套流程立起来。

#### 规则一：控制变量

一次只改**一个**因素。改了 A 又改 B，涨了 5%，你**无法归因**到底是谁的功劳——
而题目要的恰恰是"explain why you chose to incorporate each factor"。

#### 规则二：预算分级（两阶段筛选）

GPU 修好后单次 baseline 10 epoch 大约几分钟。即便如此，也不要盲跑：

| 阶段 | 配置 | 目的 |
|---|---|---|
| **筛选** | 15–20 epochs，1 个 seed | 快速淘汰没用的想法，只看相对趋势 |
| **确认** | 50–150 epochs，2–3 seeds | 对筛选出的胜者出正式数字 |

> ⚠️ **筛选阶段的陷阱**：某些改动（数据增强、强正则化、MixUp）
> 在**短训练**下看起来是**负收益**，只有在长训练下才能兑现价值。
> 判断依据：如果某个改动让 **train_acc 变低但 test_acc 没明显变差**，
> 那是"更强的正则化"，值得在长训练里再给它一次机会。
> **把这句话写进报告，是很强的加分点。**

#### 规则三：实验编号与配置留痕

每次实验给一个 ID（`exp_001`、`exp_002`...），config 必须落盘。
报告里的每个数字都要能追溯到一次具体的 run。

#### 规则四：维护一张主实验表

**现在就建**，边跑边填（用 pandas 生成 Markdown/LaTeX 表格）：

| exp_id | 改动内容 | epochs | batch | lr | optimizer | aug | test_top1 | test_top5 | Δ vs baseline | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| 000 | baseline（题目架构） | 10 | 128 | 0.01 | SGD | – | ??.?? | ??.?? | — | 参照 |
| 001 | +Normalize | | | | | | | | | |
| 002 | +BatchNorm | | | | | | | | | |
| ... | | | | | | | | | | |

> 💡 **这张表最终几乎原样进报告**（评分项 2 和 3 都要它）。
> 增量列 `Δ vs baseline` 就是题目要的 "improve accuracy **incrementally**" 的直接证据。

#### 规则五：验证集

**最佳实践**：从 50k 训练集里切出 5k 做验证集，**所有决策只看验证集**，
最后才在测试集上报一次最终数字。

**为什么**：如果你反复用测试集调参，测试集就变成了训练集的一部分，
你报出的数字是**乐观偏置**的。这是 Lab 里最常被点名的概念。

> 💡 **务实做法**：考虑到作业明确按测试集评分，
> 可以"验证集做决策 + 最后用全部 50k 重训一次"（业界标准流程）。
> 如果你决定直接用测试集调参，**在报告里主动说明这一点并讨论其局限性**——
> 这比装作没这回事要好得多，也更像研究者。

### 3.2 实验因素清单

下面按**建议尝试顺序**排列（大致按"性价比"降序）。
每个因素都给你：**原理 → 查什么 → 怎么做 → 预期 → 自检**。

> 💡 **不要全做**。挑 6–10 个因素做**扎实的**，远好过做 20 个每个都只跑一次。
> 报告要的是"深度"和"归因"，不是清单长度。

---

#### 🥇 因素 1：输入归一化（Normalize）

- **原理**：见 1.2。让优化曲面更良态，对 lr 更鲁棒。
- **做法**：`transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)`。
- **预期**：小到中等正收益，但**是后续所有技巧的地基**（尤其有 BN 时）。
- **自检**：训练集和测试集用的是**同一组** mean/std（必须一致）。

---

#### 🥇 因素 2：BatchNorm

- **原理**：对每个 mini-batch 的每个通道做标准化，再用两个**可学习**参数 `γ, β` 缩放平移。
  作用是让每层输入分布稳定、平滑损失曲面、允许更大学习率、并带一点正则化效果。
  （注：原始论文的"internal covariate shift"解释后来被质疑，实践中更站得住的说法是
  "使优化景观更平滑"。**在报告里提到这个nuance会显得你很懂。**）
- **查什么**：`nn.BatchNorm2d`；`momentum` 与 `eps` 的含义；`affine` 参数；
  **`running_mean`/`running_var` 与训练/推理行为的关系**。
- **怎么做**：每个 `Conv2d` 之后、`ReLU` 之前插入 `nn.BatchNorm2d(out_channels)`。
- **预期**：**通常是所有单项改动里收益最大的一个**，可能 +10% 以上。
- **⚠️ 坑**：BN 让 baseline 那条 "SGD lr=0.01" 的最优学习率变大（可以试 0.05~0.1）。
  加了 BN 之后**必须重新扫一下学习率**，否则你可能误以为 BN 没用。
- **⚠️ 坑**：评估时忘记 `model.eval()`，BN 会用当前 batch 的统计量而非 running 统计量，
  结果随机且偏差大。**这是加了 BN 后最常见的静默 bug。**

---

#### 🥇 因素 3：数据增强

- **原理**：人为扩大数据分布，迫使模型学"不变性"（对平移、翻转不变），从根本上抑制过拟合。
  相当于一种高效的正则化，不需要额外的模型参数。
- **查什么**：
  - `transforms.RandomCrop(32, padding=4)` —— CIFAR 的经典做法（先 pad 到 40×40 再随机裁 32×32）
  - `transforms.RandomHorizontalFlip()` —— 水平翻转
  - `transforms.RandomErasing` / Cutout —— 随机遮挡一块区域
  - `transforms.RandAugment` / `TrivialAugmentWide` —— 自动化增强策略（**注意：这是 torchvision 的
    通用增强模块，不是"标准模型"，不违反禁止条款，但要自己判断**）
  - MixUp / CutMix —— 需要改训练循环和 loss（见因素 6）
- **怎么做**：只加在训练集 transform 上。
- **预期**：Crop+Flip 是**性价比最高的组合**，中等正收益且几乎没有副作用。
- **⚠️ 坑**：**不要用 `RandomResizedCrop` 的默认参数**——它默认裁剪面积 8%~100%，
  对 32×32 的 CIFAR 会裁得过于激进（一放大就糊成马赛克），反而掉点。
  要么用 `RandomCrop(32, padding=4)`，要么把 `scale=(0.8, 1.0)` 收紧。
- **⚠️ 坑**：**垂直翻转（VerticalFlip）不要加**。CIFAR 是自然物体照片，
  上下颠倒的"狗"不存在于真实分布里，加了通常掉点。**这个对比实验很适合写进报告**：
  证明你理解"增强必须保持标签语义不变"这条原则。
- **自检**：可视化增强后的图像（1.4 节那套代码复用），确认图像仍然可辨认、
  没有把物体裁没了、颜色没有崩坏。

---

#### 🥈 因素 4：优化器

- **原理**：SGD 用统一的学习率；Adam/AdamW 用梯度的一阶/二阶矩估计给**每个参数**自适应步长；
  AdamW 修正了 Adam 中 weight decay 与自适应缩放耦合的问题（解耦权重衰减）。
- **查什么**：`torch.optim.SGD`（`momentum`、`nesterov`）、`torch.optim.AdamW`
  （`weight_decay`）、以及题目提到的 **`Muon`**（较新的优化器，可搜论文；
  注意它主要面向矩阵参数，用在 conv 上需要额外处理，属于**挑战性加分项**）。
- **怎么做**：分别试 `SGD+nesterov`、`AdamW`，**各自配合自己的最佳 lr**。
- **预期**：可能中等正收益。但注意——**优化器和 lr 是强耦合的**，
  拿 SGD 的 lr 直接套 Adam 是不公平比较。**这是控制变量法的关键细节，写进报告。**
- **⚠️ 坑**：AdamW 的 `weight_decay` 通常取 `1e-2 ~ 5e-2` 才有效，
  而 SGD 常用 `5e-4`。两者尺度不同，不要混用。

---

#### 🥈 因素 5：学习率调度

- **原理**：
  - **warmup**：训练初期模型随机、梯度方向噪声大，用小 lr 热身避免一开始就跑偏（大 batch 时尤其重要）。
  - **cosine annealing**：lr 按余弦曲线从大到小衰减，后期小步精修。
  - **OneCycle**：先升后降，利用"高温阶段大步探索 + 低温阶段收敛"。
- **查什么**：`torch.optim.lr_scheduler.CosineAnnealingLR`、
  `OneCycleLR`、`SequentialLR`（用于拼 warmup + cosine）。
- **怎么做**：加 scheduler，**注意 `scheduler.step()` 的调用时机**。
- **预期**：中等到较大正收益，且**几乎零额外成本**——这是"白捡"的分数。
- **⚠️ 坑（本章最大坑）**：`OneCycleLR` 是**按 step 更新**的（每个 batch 调一次 `step()`），
  而 `CosineAnnealingLR` 是**按 epoch 更新**的。搞混了会得到一条莫名其妙的 lr 曲线，
  训练直接崩。**每次加了 scheduler，都画一遍 lr 随 step 的变化曲线来验证**——
  这是 3 行代码的事，能省你半天 debug。
- **💡 加分动作**：把 lr 曲线图和 loss 曲线图画在同一个报告页里，说明二者的关系。

---

#### 🥈 因素 6：正则化与损失设计

- **Weight decay**：L2 惩罚，抑制权重变大。查 `optimizer` 的 `weight_decay` 参数。
  ⚠️ 通常**不该**作用于 BN 的 `γ, β` 和 bias，可以按参数分组设置。
- **Label smoothing**：把 one-hot 的 1 换成 `1-ε`，其余摊薄。
  防止模型过度自信，改善校准和泛化。查 `nn.CrossEntropyLoss(label_smoothing=...)`。
  ⚠️ CIFAR-100 每类只有 500 张训练图，标注噪声相对更值得防，这个技巧在这里尤其合理。
- **Dropout**：查 `nn.Dropout` / `nn.Dropout2d`。
  ⚠️ 在卷积网络上，放在**GAP 之后、FC 之前**通常最有效。
  注意本题架构的 FC 层只有 256→100，容量很小，dropout 收益可能有限——
  **这个"预期它没用但试了发现确实没用"的负结果，同样值得写进报告**。
- **MixUp / CutMix**：把两张图线性混合，标签也线性混合。
  强正则化，在长训练下效果显著。查 `torchvision.transforms.v2.MixUp/CutMix`。
  ⚠️ 需要改 loss（软标签）和评估逻辑。**建议放在最后有余力时再做。**

---

#### 🥉 因素 7：网络架构（重点投入方向）

这一项自由度最大，也是**最能体现"自己的设计"**的地方（题目明确鼓励
"design your own architecture ... do not import or replicate standard models"）。

**底线纪律**：只能用你自己搭的层和模块。禁止 `torchvision.models.resnet18(...)`
或任何预训练/现成分类网络。**也不能"抄"**——把 ResNet-18 的代码原样贴进来同样违规。
可以用残差连接、3×3 堆叠这些**通用组件**，但要形成你自己的设计并说明理由。

✍️ 值得尝试的方向：

**7a. 3×3 卷积堆叠 + 下采样（最优先）**

- **原理**：两个 3×3 卷积的感受野等于一个 5×5，但参数从 `2·9·C²=18C²` 降到 `25C²`
  （同样通道数下），且中间多了一次非线性。再配合 stride-2 或 max-pool 下采样，
  计算量按 `1/4` 递减。
- **查什么**：`torch.nn.MaxPool2d`、`nn.Conv2d(stride=2)`、`AdaptiveAvgPool2d`。**注意 baseline 没有下采样，全程 32×32。
  你的设计里加上下采样，可以让网络做到 4 倍深度而计算量持平。**
- **预期**：**大正收益**——这是最高效的架构改动。
- **思考题**：为什么 VGG 之后业界基本抛弃了大卷积核？写出你的理解。

**7b. 残差连接**

- **原理**：`y = F(x) + x`。恒等捷径让梯度能直接回传，缓解深层网络的退化问题，
  使网络更容易优化（注意：是"更容易优化"而非"防止梯度消失"这么简单——
  这个区分在报告里写对是加分项）。
- **查什么**：需要保证 `F(x)` 和 `x` 的**形状与通道数一致**，
  否则要用 1×1 卷积（`nn.Conv2d(C_in, C_out, 1, stride=s)`）做 shortcut 投影。
  ⚠️ **这是自己搭残差块时最常见的 bug：形状不匹配直接报错，或者你加了 stride 但没处理 shortcut。**
- **预期**：中等正收益，且**让你能训更深的网络**。

**7c. 深度可分离卷积**

- **原理**：把标准卷积拆成 depthwise（每通道独立空间卷积）+ pointwise（1×1 跨通道混合）。
  参数量和计算量降到约 `1/C_out + 1/k²`。题目明确点名了它。
- **查什么**：`nn.Conv2d(groups=...)`。**`groups=in_channels` 时就是 depthwise 卷积**——
  不需要自己写循环！这是本题的一个"顿悟点"。
- **预期**：**同等参数量下**能变深变宽，收益取决于你怎么用它。单换不涨甚至掉点，
  **必须配合"把省下来的参数加到深度/宽度上"** 才有意义。**这个 trade-off 的分析是报告的亮点。**

**7d. Stem 设计**

- **原理**：baseline 第一层是 `Conv2d(3, 256, 5)`，把 3 通道直接升到 256，
  参数占 `3×256×25+256 ≈ 19k`（其实不大），但**计算量**占不小。
  更主流的做法是"stem 用少通道 + 快速下采样"。
- **怎么做**：例如 `Conv(3→32, 3×3) → BN → ReLU`，再逐步升通道。

**7e. 归一化层替换**

- **原理**：BatchNorm 依赖 batch 统计量；某些场景下 GroupNorm/LayerNorm 更稳。
  题目 Task B 要求实现 SwitchNorm，你可以**先在 Task A 里试试 `nn.GroupNorm`**，
  为 Task B 建立直觉。这也是把两部分打通的好机会。

**7f. 容量缩放**

- **怎么做**：加宽（通道数 ×2）、加深（层数 ×2），观察**边际收益递减**在哪个点出现。
- **预期**：先涨后平。**把"深度/宽度 vs 精度"画成曲线图，是很漂亮的报告图。**
- ⚠️ 配合 GPU 显存和训练时间，别失控。

---

#### 因素 8：训练策略与收尾技巧

- **训练更久**：题目明确把"longer training duration"列为可试因素。
  长训练 + 强正则化 + lr schedule 是标准组合。
  ⚠️ 别只把 epochs 从 10 改到 100 就完事——那样后期会过拟合。
  **要成组地改**（长训练 ↔ 强增强 ↔ lr 衰减），并在报告里说明这是**一组协同**的改动。
- **EMA（指数移动平均权重）**：维护一份参数的滑动平均用于评估，几乎免费涨点。
  ⚠️ 注意 EMA 模型同样要正确切换到 `eval()` 且**不要**把它卷进梯度计算。
- **TTA（测试时增强）**：测试时对每张图做 `原图 + 水平翻转` 各预测一次，取平均。
  通常能白捡 0.5–1.5%。**查一下"为什么 TTA 有效但它不算作弊"**——
  它没有用任何测试集标签，只是对同一输入的多个视角做集成。
- **多 seed 集成**：训练 3 个不同 seed 的模型，预测概率平均。
  稳涨，但成本 ×3。**报告里说明"集成的收益 vs 成本"是成熟的工程判断。**

### 3.3 工程性能优化（省时间 = 省生命）

修好 GPU 后，再上这几个：

| 技巧 | 一行代码 | 收益 |
|---|---|---|
| **AMP 混合精度** | `with torch.autocast('cuda', dtype=torch.float16):` 包住前向+loss | 提速 1.5–2×，省显存 |
| **GradScaler** | `torch.amp.GradScaler()` | AMP 数值稳定性必需，防止梯度下溢 |
| **channels_last** | `model.to(memory_format=torch.channels_last)` | 4060 上通常有额外提升 |
| **TF32** | `torch.backends.cuda.matmul.allow_tf32 = True` | 免费加速 matmul |

> ⚠️ **AMP 的坑**：`loss.backward()` 必须用 `scaler.scale(loss).backward()`，
> `optimizer.step()` 必须用 `scaler.step(optimizer)`，且 `scaler.update()` 不能忘。
> 顺序错了会**静默地不更新参数**——loss 不降，你以为是模型问题，其实是代码问题。
>
> 🔍 **AMP 自检法**：跑 20 个 step，打印 loss。如果 loss 完全不动，先怀疑 AMP 顺序。
> 另外 `autocast` 要**包住前向和 loss 计算**，但**不要**包住 `optimizer.step()`。
>
> **报告彩蛋**：报一下"开启 AMP 前后的 epoch 时间对比"，是很实在的工程细节。

### 3.4 建议的实验路线（可直接当报告骨架）

```
exp_000  baseline（题目架构，10ep）                    → 得 baseline 数字
   │
exp_001  + 输入归一化                                   ┐
exp_002  + BatchNorm  （并重新扫 lr）                   │ 配方改良
exp_003  + lr 调度（warmup + cosine）                   │ （多数收益在这里）
exp_004  + RandomCrop + HorizontalFlip                  ┘
   │
exp_005+ 架构改良：3×3 堆叠 + 下采样 + 残差 + BN
   │      （每个架构变量单独一次实验，逐个归因）
   │
exp_0k   长训练 + 强增强 + 深网络（协同）
   │
exp_final  TTA / EMA / 多 seed 集成
```

> 💡 **报告叙事的黄金结构**：不要按时间顺序写流水账，
> 按"**假设 → 实验 → 结论 → 下一步假设**"写。
> 每次实验都回答：(a) 我为什么认为这个改动会有效？(b) 我改了什么？(c) 结果如何？
> (d) 我从中得出什么结论？(e) 因此我下一步试什么？
> **这个闭环就是 20 分的评分点。**

### 3.5 反面案例：这些"假涨点"要避免

| 现象 | 真相 |
|---|---|
| 改了 3 个东西涨了 4% | 无法归因，等于白做。拆成 3 次实验 |
| 涨了 0.3% 就宣布有效 | 在 seed 噪声内，需要多 seed 验证 |
| 用测试集反复调参到最高 | 过拟合测试集，数字不可信 |
| 换了优化器但没重扫 lr | lr 不是最优的，比较不公平 |
| baseline 和实验用了不同 epochs | 不可比（长训练天然更高） |
| 只报一个最优结果 | 应报告趋势与方差，而非单点 |

### ✅ 第 3 阶段验收

- [ ] 有一张完整的主实验表，每行可追溯到 `runs/exp_xxx/`
- [ ] **每次只改一个变量**，且每个改动都有"为什么这么改"的理由
- [ ] 至少 6–10 个扎实的因素被评估（含一些**负结果**）
- [ ] 最终精度**显著高于 baseline**（不是 0.5% 那种）
- [ ] 训练集/测试集曲线能讲出"过拟合在哪个 epoch 开始"
- [ ] 有 lr 调度曲线、增强可视化、架构示意
- [ ] 报告能说清：哪些改动收益大、哪些小、为什么

---

## 第 4 阶段：报告与提交（10 分）

### 4.1 建议的报告结构

```
1. Introduction / Task Overview
2. Environment Setup        ← 含 CPU vs GPU 的速度对比表
3. Dataset & Preprocessing  ← 统计量、增强可视化图
4. Baseline (Scoring Item 1)
   4.1 Architecture description（含参数量表）
   4.2 Training setup（超参表）
   4.3 Results（曲线 + top-1/top-5）
5. Experiments & Improvements (Scoring Item 2) ★
   5.1 Experimental methodology（控制变量、预算分级、验证集策略）← 强调你懂方法论
   5.2 Factor 1: ...（动机 / 配置 / 结果 / 结论）
   5.3 Factor 2: ...
   ...
   5.k Ablation summary table（主实验表） + 柱状图
6. Final Model & Error Analysis
   6.1 最终配置
   6.2 混淆矩阵 / per-class accuracy
   6.3 典型错误样例分析
7. What I Learned (Scoring Item 4)
8. References
```

**评分项 3 的具体得分点**（"well-organized / tables / high-quality figures"）:

- 所有超参都用**表格**呈现，不要散落在正文
- 所有实验对比都用**表格 + 图**双重呈现
- 图的分辨率足够（`dpi=150+`），坐标轴有 label，有图例，多种线型**在黑白打印下也能区分**
- 图表**编号并引用**（"as shown in Table 3"）
- 页眉页脚统一，公式用数学模式写

> 💡 **强烈建议用 LaTeX**（题目说"it would be better"）。
> 即使你没用过，一个最简 `article` 文档 + `booktabs`（表格）+ `graphicx`（插图）
> 就能满足要求。**LaTeX 画 Booktabs 风格的三线表比 Word 好看一个量级**，
> 这是 5 分里最容易拿的部分。

### 4.2 图表清单（照着做就不会漏）

| 图 | 类型 | 说明 |
|---|---|---|
| 数据样例网格 | 图片网格 | 反归一化后的训练样本 + 标签 |
| 增强效果对比 | 图片网格 | 同一张图的 N 次随机增强，展示多样性 |
| 训练曲线 | 折线 | train/test loss 与 acc 双图，标注过拟合起点 |
| lr 调度曲线 | 折线 | 验证 scheduler 配置正确 |
| 架构示意 | 自绘 | 用 `torchinfo` 表代替也可以 |
| 消融对比 | 柱状图 | 各因素的 Δ 精度，按收益排序 |
| 混淆矩阵 | 热力图 | 最终模型，100×100（可用超类聚合） |
| per-class accuracy | 条形图 | 找出最难/最易的类 |
| 错误样例 | 图片网格 | 预测错的图 + 真实/预测标签 |

> 💡 **混淆矩阵和 per-class 分析是"超越了刷分"的信号**。
> CIFAR-100 有 20 个超类（superclass），你会发现错误大量发生在**同一超类内部**
> （比如各种花、各种鱼之间混淆）。**这个观察非常值得写进报告**，
> 它说明模型学到了语义结构，而不是随机犯错。

### 4.3 "What have you learned"（5 分）怎么写

**不要**写"我学会了 PyTorch 和 CNN 的基础知识"这种空话。

**要**写**具体的、有证据的**认知转变。可以围绕：

- 你原本以为 X，实验后发现是 Y。（例：以为加深网络必然涨点，实测发现没有配套的
  BN + 残差连接时深网络反而更差）
- 你踩过的**具体**坑，以及事后理解的**根因**。（例：忘了 `model.eval()` 导致的指标异常）
- 你对"实验方法论"的体会。（例：为什么单次实验的 0.3% 提升不可信；
  为什么控制变量比堆技巧更重要）
- 你对某个概念的理解如何被实验修正。（例：BatchNorm 的真实作用机制）
- 如果能重做，你会怎么改流程。

> 💡 这 5 分考的是**反思的诚实度和具体性**。写 3–5 个真实的、带细节的认知点，
> 远好过 10 条泛泛而谈。**负结果和失败经历是最有价值的素材**——
> 题目说了要"evidence of your effort"，失败的分析恰恰是最好的证据。

### 4.4 提交清单

- [ ] **PDF 报告**（英文！）单独上传
- [ ] **代码 zip** 单独上传，**不含报告**
- [ ] notebook 已清理：无报错残留输出、有 markdown 说明、有必要注释
- [ ] notebook **从头 Execute All 能跑通**（这条极其重要，务必验证一次）
- [ ] `runs/` 里的日志/图表是否要打包？（建议保留关键的 config + metrics，体积小且是证据）
- [ ] 大文件（`data/`、`*.pt` 权重）**不要**放进 zip
- [ ] 报告里的每个数字都能在代码/日志中找到出处

> ⚠️ **"从头跑通"这条最容易翻车**：你在 notebook 里可能不自觉依赖了
> 某个只在上一步执行过的变量、或某个已被覆盖的全局状态。
> 交之前一定要 **Kernel → Restart & Run All** 完整跑一遍。
> 报错的 notebook 会严重影响"clean and well-organized"这一项的评分。

---

## 附录 A：高频陷阱清单（对照自查）

| # | 陷阱 | 症状 | 解法 |
|---|---|---|---|
| 1 | 装了 CPU 版 torch 却在跑"GPU"实验 | 慢得反常识 | 见 0.1 节 |
| 2 | 模型末尾又加了 Softmax | loss 几乎不降，精度极低 | `CrossEntropyLoss` 已含 log_softmax |
| 3 | 评估忘 `model.eval()` | 指标随机波动（加 BN 后） | 训练/评估模式严格切换 |
| 4 | 评估没包 `torch.no_grad()` | 显存持续增长最后 OOM | 包住整个评估循环 |
| 5 | 忘了 `zero_grad()` | loss 不降或震荡 | 训练循环五步顺序 |
| 6 | `accuracy` 累积时没 `.item()` | 显存缓慢泄漏 | 每个 batch 立即转 Python 数 |
| 7 | 测试集也做了随机增强 | 指标震荡、虚高 | 测试 transform 只留 ToTensor+Normalize |
| 8 | 归一化统计量算错维度 | 涨点很少甚至掉点 | 沿 `(0,2,3)` 求，除以 `N*H*W` |
| 9 | 用测试集算 mean/std | 数据泄漏 | 只用训练集 |
| 10 | BatchNorm 后没重扫 lr | 误判 BN 无效 | 加 BN 后 lr 可提高 5–10 倍 |
| 11 | scheduler step 频率错 | lr 曲线诡异，训练崩 | 画 lr 曲线验证 |
| 12 | AMP 顺序错 | 参数静默不更新 | scaler.scale/step/update 三件套 |
| 13 | 残差块维度不匹配 | 运行时 shape 报错 | shortcut 用 1×1 conv 投影 |
| 14 | Windows `num_workers>0` 卡死 | 训练挂起 | Notebook 里用 0 |
| 15 | 用了 torchvision 预置模型 | **违反题目规则** | 只用自己的模块 |
| 16 | 只报最佳单次结果 | 报告不严谨 | 报均值±std 或说明方差 |
| 17 | 图表 dpi 太低 | 报告显得粗糙 | `dpi=150` 以上 |
| 18 | 交前没 Run All | notebook 报错 | Restart & Run All |

---

## 附录 B：每日自查问题（检验你是否真的"学会"了）

能不看笔记回答出来，就说明掌握了：

1. 训练循环为什么是 `zero_grad → forward → loss → backward → step` 这个顺序？
2. `CrossEntropyLoss` 内部做了什么？为什么它和"Softmax + NLLLoss"等价？
3. 为什么归一化要自己算 mean/std 而不用测试集的？
4. BatchNorm 在训练和推理时的行为有什么本质区别？`running_mean` 起什么作用？
5. 为什么"两个 3×3"优于"一个 5×5"？给出参数量和感受野的具体计算。
6. 数据增强为什么能提升泛化？为什么垂直翻转对 CIFAR 有害？
7. 学习率 warmup 解决什么问题？为什么大 batch 时更需要它？
8. 为什么"涨了 0.3%"不能下结论？你需要做哪些事才能下结论？
9. 你的实验里有没有"同时改了两个变量"的情况？怎么补救？
10. 如果让你把这个模型再提升 2%，下一步你会先试什么？为什么？

---

## 附录 C：关键 API 速查（自己去读官方文档，别只抄）

| 用途 | 查什么 |
|---|---|
| 数据集 | `torchvision.datasets.CIFAR100` |
| 变换 | `torchvision.transforms.Compose / ToTensor / Normalize / RandomCrop / RandomHorizontalFlip / RandomErasing` |
| 数据加载 | `torch.utils.data.DataLoader` |
| 层 | `nn.Conv2d / BatchNorm2d / GroupNorm / ReLU / GELU / Linear / Dropout / AdaptiveAvgPool2d / MaxPool2d / Flatten` |
| 损失 | `nn.CrossEntropyLoss`（注意 `label_smoothing`） |
| 优化器 | `torch.optim.SGD / AdamW` |
| 调度器 | `torch.optim.lr_scheduler.CosineAnnealingLR / OneCycleLR / SequentialLR / LinearLR` |
| 精度加速 | `torch.autocast` / `torch.amp.GradScaler` |
| 模型摘要 | `torchinfo.summary` |
| 权重平均 | `torch.optim.swa_utils.AveragedModel` |

---

## 建议的时间安排（距截止 2026-10-21 约 6 天）

| 日期 | 任务 |
|---|---|
| D1 | 第 0 阶段（**先修 GPU**）+ 第 1 阶段数据管线 |
| D2 | 第 2 阶段 baseline 跑通、归档、出曲线 |
| D3–D4 | 第 3 阶段：配方改良（归一化/BN/lr/增强）+ 架构改良，边跑边填实验表 |
| D5 | 收尾技巧 + 最终模型训练 + 错误分析 |
| D6 | 写报告 + LaTeX 排版 + Run All 验证 + 提交 |

> ⚠️ **缓冲建议**：D3–D4 的实验很容易超时（尤其是架构探索）。
> **先跑"性价比高"的因素**（BN、lr 调度、增强），把高分数的增量先锁定，
> 再去做有风险的架构实验。这样即使时间不够，你也有完整的实验链。

---

**最后一句**：题目说 "We highly encourage you to explore your own designed solution."
不要把它当成"跑一遍主流 trick 清单"。**真正的得分点是你能不能把每一步
都讲成一个有假设、有验证、有归因的推理过程。** 祝顺利。
