# 香港中文大学（深圳）· 数据科学学院

# AIR5011 / MDS5122 —— Assignment 1

**截止时间：2026 年 10 月 21 日 23:59**

---

## 说明（Instructions）

1. 你必须在 Blackboard 上提交作业。请上传一个 PDF 文件以及你的代码。
   **PDF 文件与代码（以 zip 压缩包形式）须分别上传**，且不要把报告放进 zip 压缩包内。
2. 你的提交需要结构清晰、呈现良好才能获得满分。请详细说明**你如何运用所学知识来解决问题**，
   并**分步骤**展示你提出的解决方案。确保你的解答易于理解，并且用**英文**撰写。

---

## A. 使用 PyTorch 搭建神经网络（40 分）

要实现一个深度学习应用，最简便的方式是使用现代深度学习框架来编写代码。学术界最流行的框架是
[PyTorch](https://pytorch.org/)。在第一部分中，我们将使用一个相对较小的数据集——[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
（50,000 个训练样本、10,000 个测试样本，每张为 32×32 的彩色图像）——来开发一个图像分类应用，
并试验各种因素以提升测试集上的性能。你的代码必须使用 **PyTorch（2.0 或更高版本）**和
[Jupyter Notebooks](https://jupyter.org/) 编写。Notebook 中应包含你的实验结果和必要的注释。

你可以尝试不同的网络设计（例如残差连接、3×3 卷积堆叠）、不同的组件（例如深度可分离卷积）、
不同的优化器（例如 AdamW 和 Muon）、不同的训练方案（例如 warmup 和线性学习率调度器），
或者任何你能想到的、有助于提升精度的其他因素；你可以设计自己的架构并加入残差连接等组件，
但**不要从 torchvision 或其他库中导入或复刻标准模型（如 ResNet-18）**。
我们非常鼓励你探索自己设计的解决方案。

精度在测试集上评估，且越高越好。但是，我们**不会**仅根据所达到的精度来给分；相反，
我们寻求的是**你付出努力、并多次尝试逐步提升精度**的证据。你必须在实验报告中详细记录
为提升精度所采取的步骤，包括你考虑过和评估过的因素，并解释你为何选择纳入每一个因素。
此外，你还应在提交中加入以下部分，以获得相应的分数。

### 评分标准（Scoring Criteria）

1. 搭建 Python 编程环境，编写如下所示的简单 CNN，在 CIFAR-100 数据集上训练 **10 个 epoch**，
   并报告其精度；该精度将作为 baseline（基线）。（**10 分**）

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

2. 你考虑了哪些因素来提升精度？例如，你可以尝试更深、更宽的卷积神经网络，更好的优化器、
   数据增强方法、dropout 层，或者更长的训练时长。这些只是一些建议，你同样可以探索其他方案。
   我们期望你的最终精度高于 baseline。（**20 分**）

3. 你的代码和提交文档应当整洁且组织良好。例如：段落组织有序；结果以表格形式呈现，
   并通过高质量的图表得到有效可视化；如果文档使用 **LaTeX** 撰写会更好。（**5 分**）

4. 通过这次作业的尝试，你学到了什么？（**5 分**）

---

## B. 从零开始搭建神经网络（60 分）

为了更好地理解神经网络究竟是什么，你需要在**不使用任何深度学习库**的情况下，编写、训练并评估
你自己的深度神经网络。你**禁止**使用任何深度学习框架或自动微分库，例如
[PyTorch](https://pytorch.org/)、[TensorFlow](https://www.tensorflow.org/) 和
[JAX](https://jax.readthedocs.io/en/latest/)；相反，你只应使用 [NumPy](https://numpy.org/)
来实现这些功能。如果你发现性能不理想，你应该用**向量化**方式实现这些函数以提升速度。

### 注意事项（Notice）

1. 你应当为每个网络模块实现**前向传播和反向传播**，确保它们与 PyTorch 的实现方式相似；
   例如，模块应实现为一个 Python 类。
2. 下面展示的某些网络模块标注了 *config*，表示它们在初始化时需要用到这些参数。
   这些名称的含义请参考 [PyTorch 的文档](https://pytorch.org/docs/stable/index.html)。
   你的实现应当与 PyTorch 的实现高度一致。

### 评分标准（Scoring Criteria）

1. 实现以下网络模块：（**40 分**）

   a) **GELU**，使用 tanh 近似（**5 分**）

   $$\mathrm{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\tfrac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$

   b) **Linear**，config：*in_features, out_features, bias*（**5 分**）

   c) **CrossEntropyLoss**（**5 分**）

   d) **Adam**，config：*lr*（**5 分**）

   e) **Conv2d**，config：*in_channels, out_channels, kernel_size, stride, bias*（**10 分**）

      **你绝对不能使用任何已有的卷积函数。**

   f) **SwitchNorm2d**，参考[论文](https://arxiv.org/abs/1907.10473)（arXiv:1907.10473），
      config：*num_features*（**10 分**）

2. 使用你构建的模块，开发一个简单的 CNN，在 CIFAR-100 数据集上训练网络。
   将其性能与 PyTorch 的对应实现进行比较，以评估精度是否相近。（**10 分**）

3. 你的代码和提交文档应当整洁且组织良好。例如：段落组织有序；结果以表格形式呈现，
   并通过高质量的图表得到有效可视化；如果文档使用 **LaTeX** 撰写会更好。（**5 分**）

4. 通过这次作业的尝试，你学到了什么？（**5 分**）

---

## 环境搭建指南（Environment Setup Guide）

1) **安装 Miniforge**
   - ✅ 访问 https://github.com/conda-forge/miniforge?tab=readme-ov-file#install
   - ✅ 按照说明在你的系统上安装它

2) **打开命令提示符（Command Prompt）**
   - ✅ 按 `Windows + R`，输入 `cmd`，然后按回车

3) **创建 Python 环境**
   - ✅ 输入：`conda create -n myenv python=3.10 -y`
   - ✅ 然后输入：`conda activate myenv`

4) **安装 PyTorch**
   - ✅ **仅 CPU 版本**（适用于所有电脑）：
     - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
   - ✅ **NVIDIA GPU 版本**（仅在你拥有 GPU 时）：
     - 前往 https://pytorch.org，选择你的 CUDA 版本，然后复制对应的 pip 命令

5) **安装额外工具**
   - ✅ 输入：`pip install matplotlib ipykernel`

6) **安装 VS Code**
   - ✅ 从 https://code.visualstudio.com/ 下载
   - ✅ 打开 VS Code，安装 "Python" 扩展（由 Microsoft 提供）
   - ✅ 打开一个 `.py` 文件，并在右下角选择 Python 解释器（myenv）

7) **测试**
   - ✅ 创建一个文件 `test.py`，内容为：
     ```python
     import torch
     print(torch.__version__)
     ```
   - ✅ 在 VS Code 中或命令提示符中运行它（需激活 "myenv"）。你应该会看到类似 `2.5.0` 的版本号。

如果你遇到任何问题，请随时在班级微信群中提问。

---

> 译自 `Assignment_1.pdf`（AIR5011 / MDS5122，The Chinese University of Hong Kong, Shenzhen · School of Data Science）。
> 原文中高亮部分（提交方式、英文撰写要求）在译文第 1 条说明中保留强调。
