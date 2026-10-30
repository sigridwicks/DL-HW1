# HANDOVER — AIR5011 / MDS5122 Assignment 1 · Task A + B

> 2026-10-19 更新（**A 与 B 的实验/代码全部结束；Part A 报告已写、Part B 草稿已备**）。历史细节见记忆文件、`task_a/runs/`、`task_b/runs/`，本文件只留当前必要的。
> **截止 2026-10-21；剩余工作只有「报告收尾 + 提交检查」。**
> 面向接手此项目的下一个 agent 会话。与用户沟通用**中文**；指代 cell 一律用 **`code #N`（只看代码格、从 1 数）**。

## ⚠️ 先读这段：当前状态与第一小时

**两个任务都不要再跑新实验。** Part A 的最终模型已交付（50k 重训、test10k 只测一次）；Part B 的六个模块、整网对拍、端到端训练与 torch 参照**全部完成并验收**（见 **§10**）。

- **Part A headline（test10k）**：单模型 **75.01 ± 0.32**；3-seed 集成 **77.57**；**集成 + TTA 77.87（top-5 94.36）**。
- **Part B headline**：numpy 端到端 val **42.3%** vs torch **42.0%**（Δ 0.24 点）；L2 整网对拍 **1.78e-15**；每个 epoch 慢 **5.1×**。
- **test 已冻结**：从 2026-10-17 起不再改动模型、不再用 test 做任何选择。再调参就是任务书 3.5 的反面案例。

**下个 agent 的第一小时，按这个顺序：**
1. 读本文件 §0–§3 + **§10（Part B）**（15 分钟），再读 `TaskA_任务书.md` 的 3.1–3.5、4.1–4.4 与 `TaskB_任务书.md` 的 4.1/4.2/4.4。
2. 读 §9「报告写作地图」——它把报告的每一节对到具体的数字/图/文件。
3. 打开 `report/main.tex`：**Part A 已完整**（摘要 + 7 节）；Part B 目前只是骨架 → 用 `report/part_b_draft.tex` 替换第 352 行 `\clearpage` 之后的全部内容。
4. 和用户确认两件事：**B.5「What I Learned」由谁写**（草稿里只留了 6 条英文要点）、**PDF 用什么引擎编译**（本机没装 LaTeX）。
5. 之后的顺序：报告收尾 > 提交前检查。不要主动开新实验。

## 0. 速览

| 项 | 值 |
|---|---|
| 任务 | CIFAR-100：复刻 baseline（10 分）→ 提分实验（20 分，主战场）→ 报告（10 分） |
| 必读 | `TaskA_任务书.md`（评分点 / 五个实验规则 / 因素清单 / 坑位 / 验收清单）；`TaskB_任务书.md`（六模块规格 / 对拍协议 / 报告结构）；原文 `Assignment_1.pdf`。**Task B 已完成**（见 §10）：`task_b/npnn/` 是交付实现，`task_b/tests/` 是验证工具 |
| 解释器 | `.venv\Scripts\python.exe`（3.11.14）；torch 2.11.0+cu128 / torchvision 0.26.0+cu128 |
| 硬件 | RTX 4060 Laptop 8GB；**快态**下 cnn_down@45k ≈ 25 s/epoch、cnn_wide ≈ 26、20.7M 模型 ≈ 60 |
| 数据 | `data/cifar-100-python/`（hf-mirror 下载，md5 `eb9058c3a382ffc7106e4002c42a8d85` 已校验） |
| Notebook | `02_main_exps_1.ipynb`（第一批 exp_101–109）/ `02_main_exps_2.ipynb`（第二批 exp_2xx，**code #1–#11**）/ `03_final_analysis.ipynb`（**唯一出主表处**，扫 `task_a/runs/`）/ `04_report_assets.ipynb`（报告图：benchmark、消融图、混淆矩阵、错误样例） |
| 数据与产物 | **`task_a/runs/<name>/`**（45k/5k：config + metrics + curves + model.pt）、**`task_a/runs_final/`**（50k→test、错误分析）、**`figs/`**（报告插图）。**不要凭记忆报数字** |
| 外部参考 | 另一个 AI 的方案文档（`CIFAR100最佳模型查询_Export_*.md`）**已被用户删除**；其中不能照抄的结论留档在 §3.4 |

## 1. 代码骨架 `task_a/src/`

- `config.py`：路径 / SEED=42 / CIFAR100 mean+std。
- `data.py`：`build_loaders_val(batch, num_workers, train_tf)` → (train45k, val5k, test10k)；split 用独立 Generator（不动全局 RNG）；num_workers>0 自动开 `persistent_workers`。
- `models.py`：题目 baseline 三兄弟 + `cnn_down` + **`cnn_down_variant()` 工厂**（widths / strides / convs_per_stage / residual / dw_sep / norm 六个旋钮，**不传参 = `cnn_down` 本身**）+ 12 个 wrapper：`cnn_residual` `cnn_wide`(9.36M) `cnn_deep` `cnn_3stage` `cnn_dw` `cnn_dw_wide` `cnn_groupnorm` `cnn_deep_residual`(5.19M) `cnn_deep_residual_wide`(20.7M) **`cnn_deep_residual_d3`(7.93M) `cnn_deep_residual_d4`(10.66M) `cnn_wide3`(21.0M)** ← 后三个是容量阶梯。
- `engine.py`：`fit(..., ema_decay=None)`（scheduler 按 epoch step）；**EMA 钩子**（`_EMA`）：给了 `ema_decay` 就每 step 更新影子权重、每 epoch 用 EMA 权重多评估一次，结果进 `history["ema_test_acc"]`，**不改训练也不改 model.pt**；`evaluate`（top-1）；**`evaluate_topk`**（top-1 + top-5）；**`ensemble_eval(models, loader, device, tta=False)`**（概率平均，可做 TTA / 3-seed 集成 / 两者叠加）。
- `data.py`：`build_loaders_val`（45k/5k 协议）+ **`build_train_test_loaders`**（**50k 全量 train + 10k test + 2k monitor**，S5 最终重训专用）。
- `experiments.py`：`run_experiment` / `run_batch(..., skip_existing=True)`（断点续跑）/ `build_experiment_metadata()`（**产出的 config 字段名是主表契约，勿改**；`run_extra` 是它的旧别名）/ **`run_final_retrain`**（50k 全量重训 → test 只测一次，产物写 **`task_a/runs_final/`**，不混进 `task_a/runs/`）。**`run_experiment` / `run_batch` 的 `ema_decay` 默认 0.999**（= 从现在起所有新 run 都记 EMA 曲线；单项写 `"ema_decay": None` 关掉）。
- `utils.py`：`set_seed` / `save_run`（四件套；有 EMA 时 metrics.csv 多两列 `ema_test_loss`/`ema_test_acc`，curves.png 多一条虚线 —— **老 run 仍是 6 列，读表按列名取，别按下标**）/ `check_gpu`（15 s 自检）/ `warmup_cosine`。

## 2. 已有结果

### 2.1 第一批（10 ep · 3 seeds · 45k/5k val）

| exp | 配方 | val | 备注 |
|---|---|---|---|
| 101 | baseline + SGD 1e-2 | 11.33 ± 0.30 | 欠训练、稳定 |
| 102 | 换 Adam | 21.07 ± 17.90 | **seed 42 坍缩到 0.76%** |
| 103 | +BatchNorm | 47.17 ± 0.25 | 坍缩治愈，波动 17.9→0.25 |
| 104 | +warmup/cosine | **54.09 ± 0.11** | 无增强最优 |
| 105 | +crop+flip | 49.93 ± 0.18 | −4.16（短预算正则化陷阱） |
| 106 | 3×3 堆叠 | 54.25 ± 0.43 | ~138 s/epoch |
| 107 | **cnn_down** | **61.00 ± 0.68** | 10ep 冠军；25–28 s/epoch |
| 108 | cnn_3x3 无增强 | 57.86 ± 0.17 | |
| 109 | cnn_down 无增强 | 59.29 ± 0.63 | **train acc 96.5%** ⚠ 10ep 就背完训练集 |

### 2.2 第二批 S1/S1b（20 ep · aug ON · 锚点 = exp_200 的 3 seeds = 67.63）

| exp | 单变量改动 | val | Δ | seeds |
|---|---|---|---|---|
| **202** | **宽度 ×2** | **70.40**（70.38/70.42/70.40） | **+2.77** | 3 ✅ |
| 208 | 深度 ×2 + 残差 | 69.08（**3-seed 复核 69.15 ± 0.50**） | +1.45 → **复核 +1.52** | 1 → 3 ✅ |
| 200 | 底座 cnn_down | 67.63（68.66/66.70/67.52） | — | 3（锚点） |
| 201 | +残差 | 67.40 | −0.23 | 1 |
| 206 | depthwise + 宽度回投 | 65.00 | −2.63 | 1 |
| 203 | 深度 ×2 | 64.52 | −3.11 | 1 |
| 204 | 去 4px 阶段 | 63.12 | −4.51 | 1 |
| 205 | depthwise | 54.58 | −13.05 | 1 |
| 207 | BatchNorm→GroupNorm | 53.92 | −13.71 | 1 |

**四条结论**：① **宽度是唯一的真杠杆**（train acc 也最高 → 真·容量不足）；② 深度单独 −3.11，**"深度 + 残差"翻正 +1.45 → 残差解锁深度**（3-seed 复核 +1.52，✅ 已坐实）；③ depthwise / GroupNorm 是灾难级负收益；④ 两个外来说法被推翻："砍 4px 阶段"（−4.51）与"depthwise 省下的参数回投宽度"（回投后仍 −2.63）。
⑤ 复核的价值：202 单 seed 是 71.42（+3.79），3 seeds 掉到 70.40（+2.77）—— **报告用复核值**。

### 2.3 第二批 S2（60 ep · 3 seeds · 2026-10-17 过夜批，15 h 13 min，无慢态、无坍缩）

| exp | 配置（除注明都带 aug） | val | Δ vs 同族 | 备注 |
|---|---|---|---|---|
| **216** | `dr_wide` 宽 ×2 + 深 ×2 + 残差 | **74.13 ± 0.2** | +1.65 vs 212 | **冠军**（全项目最高） |
| 219 | `wide3` 宽 ×3 | 73.40 ± 0.4 | +0.92 vs 212 | 21.0M |
| 220 | `wide` + Cutout | 73.09 ± 0.4 | +0.61 vs 212 | 3/3 种子同向，< 噪声底 |
| 212 | `wide` 宽 ×2 | 72.48 ± 0.3 | — | |
| 217 | `dr_d3`（3 残差块/stage） | 71.98 ± 0.1 | +0.52 vs 214 | |
| 218 | `dr_d4`（4 残差块/stage） | 71.99 ± 0.2 | +0.01 vs 217 | **深度饱和** |
| 214 | `dr` 深 ×2 + 残差 | 71.46 ± 0.2 | +13.33 vs 213 | |
| 211 | `wide` **无增强** | 61.47 ± 0.2 | — | train 100% @ep20，val 峰 ep34 |
| 213 | `dr` **无增强** | 58.13 ± 0.0 | — | train 100% @ep30，val 峰 ep43 |

**六条**：
1. **★ aug 在长预算下彻底翻案：+11.01（wide）/ +13.33（dr）**。反超点 **ep8 / ep7**，之后一路领先再没被追回。机制：无增强模型 ep20/ep30 就把 train 背到 100%，val 在 ep34/ep43 见顶回落；带增强到 ep50–60 才到 100%，val 一路涨到 ep55–58。**第一批"短训练下 aug 负收益"（10ep −4.16/−3.61）到这里闭环** —— 同一变量、符号由预算长度决定。这是报告最强的一条。
2. 冠军 **216 = 74.13**（20ep 冠军 70.40 → 60ep 74.13）；两个赢面叠加基本可加（+1.65 深度残差、+0.92 宽度、+0.61 Cutout）。
3. **深度 d3 之后饱和**（71.46 → 71.98 → 71.99）；**宽度 ×2→×3 仍 +0.92**。**同 ~21M 参数，深度+残差 > 宽度**（216 20.70M 74.13 vs 219 21.00M 73.40，+0.73）；参数效率上 217（7.93M）71.98 只比 212（9.36M）72.48 低 0.5。
4. 过拟合起点：无增强 ep34（wide）/ ep43（dr）见顶；带增强 60ep 内无回落（best≈final）。⚠️ caveat：60ep cosine 末段 lr→0，"best≈final"部分是调度造成的，只能写"60ep 内无明显回落"。
5. **EMA = 稳定器不是增益**：中段领先 raw **+6~7 pt**（212 ep20：66.0 vs 72.2；216 ep20：66.8 vs 74.0），ep60 完全相同（cosine 末段 raw 自己收敛）。EMA 曲线在 ep20–30 就达到最终水平。
6. Cutout +0.61（+0.34/+1.06/+0.42，全为正），train acc 98.7% 最低 → 确实正则化，但幅度在噪声底内。

### 2.4 60 ep 底座锚点补齐（exp_221，2026-10-17）

`exp_221_base_down60`（cnn_down + aug + 60ep，3 seeds）= 69.20 / 69.16 / 69.84 → **69.40 ± 0.3**（20 ep 的 67.63 → 长训练本身 +1.77）。补齐后两条阶梯与可加性都能算了：

| 轴 | 各点（val） | 增量 |
|---|---|---|
| 宽度（深度 1 块、无残差） | 69.40 (×1, 2.35M) → 72.48 (×2, 9.36M) → 73.40 (×3, 21.0M) | +3.08 / +0.92（递减，未饱和） |
| 深度（宽 ×1、带残差） | 69.40 (d1) → 71.46 (d2, 5.19M) → 71.98 (d3, 7.93M) → 71.99 (d4, 10.66M) | +2.06 / +0.52 / **+0.01（d3 后饱和）** |

- **可加性**：69.40 + 3.08 + 2.06 = 74.54，实际 216 = 74.13 → 残差 **−0.41**（噪声内）→ 两个赢面基本可加、略有重叠。
- **增强 vs 架构**：aug 在 wide 上 +11.01，最大的架构收益（216 vs base）+4.73 → **增强约为任何架构改动的 2.3 倍**；无增强的 wide（61.47）甚至打不过带增强的底座（69.40）。
- **评估侧（val5k，非 test）**：216 单模型 top5 = 92.98；单模型 TTA 74.84 / 93.22；**3-seed 集成 76.26 / 93.80**；集成 + TTA **76.34 / 94.08**。
- ⚠️ **归因更正**：exp_221 的 8.9 s/epoch vs S1 那批 23.8 s，是 **NUM_WORKERS 0 → 2** 的差别，**不是 GPU 慢态**（8.9 s ≈ 纯 GPU 下限 22.6 ms/batch × 351）。S1 screen 批（17:34–19:03）跑在 nw=0 上，20:39 之后的 confirm 批起才是 nw=2。数值不受影响，但报告工程段应写"2.6× 提速在 run 日志里直接可见"。

### 2.5 ★ 最终交付：50k 全量重训（exp_230，2026-10-17，`task_a/runs_final/`）

冠军配方 `dr_wide` + crop+flip + 60 ep，训练集 45k→**50k 全量**（5k val 并入），**test10k 只在训练结束后测**。3 seeds、每 seed 60 ep、66–67 s/epoch，总墙钟 2 h 15 min。

| 评估方式 | 45k/5k val（3 seeds） | **50k → test10k** |
|---|---|---|
| 单模型 | 74.13 ± 0.20 | **75.01 ± 0.32**（74.63 / 74.98 / 75.41） |
| 单模型 + TTA | 74.84（seed0） | 75.69 ± 0.33 |
| 3-seed 集成 | 76.26 | **77.57** |
| 3-seed 集成 + TTA | 76.34 | **77.87**（top-5 94.36） |

- **头号数字：77.87 / 94.36**（3-seed 集成 + TTA）。集成 +2.56、TTA +0.68（叠加后 +2.86）。
- **50k 重训比 val 估计高 +0.87**（三个 seed 都是正的：+0.43 / +1.06 / +1.13）→ 45k val 是偏保守的估计，没有过拟合验证集。
- 训练过程正常：train acc 16%→100%，monitor（train 子集、无增强）26.8%→100% 平滑上升；三个 seed 的 test 只差 0.8 pt。
- ⚠️ **test 到此冻结**：4 种评估方式都是事先声明的，没有拿 test 做任何选择。报告里四个数**全部列出**，主推 77.87，不要只挑最高的。

### 2.6 报告素材（S6，2026-10-17；图在 `figs/`）

| 项 | 数字 |
|---|---|
| CPU vs GPU（`cnn_down`, bs=128, fwd+bwd+step） | 185 ms/step（691 img/s）vs 27 ms/step（4838 img/s）→ **7.0×** |
| AMP（`dr_wide`, bs=128） | 164 → 87 ms/step（**1.89×**）；峰值显存 1379 → 861 MB（**−518 MB**） |
| 最终模型错误分析（3-seed 集成, test10k） | top-1 **77.57** / top-5 **94.12**（与 S5 summary 一致，交叉验证通过）；2243 张错分 |
| **超类内错误占比** | 812 / 2243 = **36.2%**（随机约 4%）→ 错误集中在语义相近的超类内部 |
| 最难 5 类 | boy 48%、girl 53%、otter 54%、bowl 57%、seal / possum 59% |
| 最易 5 类 | tractor 97%、motorcycle 96%、plain / wardrobe / orange 94% |
| 最常见混淆对 | boy→man 18、oak_tree→maple_tree 16、girl→woman 16、otter→seal 13、maple_tree→oak_tree 13、willow_tree→oak_tree 12 |

图：`figs/fig_ablation_60ep.png`（架构 Δ + 增强效应）、`fig_capacity_scaling.png`、`fig_confusion_fine.png`、`fig_confusion_coarse.png`、`fig_per_class.png`、`fig_error_samples.png`。
错误样例里的典型模式：同超类近义类（boy↔man、oak↔maple、otter↔seal、bowl↔plate/cup、tulip↔rose）+ 颜色主导的错（bowl→orange、sweet_pepper→orange）。

## 3. 关键机制与报告素材

1. **无 BN 的种子坍缩**：Adam + 无归一化时可能整段坍缩成常数预测器，是否发生看 seed×batch 顺序抽签；BN 把 seed 42 从 0.76% 救到 47.04%。这不是事故是证据。健康门：epoch-1 train_loss ≥ ~4.55 即可疑。
2. **★ aug 的收益取决于"模型有没有开始背题"→ 进一步取决于预算长度**：10ep 的架构×增强 2×2 里 aug 符号翻转（baseline+BN −4.16 / 3×3 −3.61 / cnn_down **+1.71**，翻转点 = train acc 冲到 96.5% 处）；到 60ep，**同一个变量变成 +11.0 / +13.3，反超点 ep7–8**（§2.3 ①）。**这是报告的核心故事**，也是 S1 筛选必须带增强的依据。
3. **★ 噪声底 ≈ 1 pt（2026-10-16 晚实测）**：`exp_202_wide_s0`（screen 71.42）vs `exp_202_wide_conf_s0`（confirm 70.38）—— 同配置、同 seed、同初始化、同数据顺序，差的 1.04 pt **全部来自 DataLoader worker 的增强 RNG**（`set_seed` 管不到 worker，`persistent_workers=True` 让流一直往下走）。推论：**Δ < 1 pt 不下结论**（201 的 −0.23 是"无影响"）；3-seed 的 std 会**系统性低估**不确定度（confirm 的 ±0.02 不能当不确定度写）。
4. **工程视角**：GPU 占用率只有 41%、功耗 43W 的真相是单进程取数 —— nw=0 58.9 ms/batch → nw≥2 22.8 ms（贴住纯 GPU 下限 22.6），**2.6× 加速**；AMP 在这个场景无效。
5. **GPU 快/慢双态**：慢态可到 14×。防线 = 性能模式 + 15 s 自检 + `run_experiment` 慢态警告。⚠️ 注意 26 vs 42 s/epoch 那种 1.7× 的差别**更可能是 num_workers 0 vs 2**（见 §2.4），别一律归给慢态：
   ```python
   import torch, time
   a = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize(); t = time.time()
   for _ in range(8): a @ a
   torch.cuda.synchronize(); print((time.time()-t)/8*1000, "ms/matmul")   # 快 ~20–40 / 慢 ~300
   ```
6. **报告方法论分（已定的设计决定）**：星型消融而不是贪心（每个 Δ 可归因）；两级预算（1 seed 筛 → 3 seeds 确认）并**如实报告筛选偏差 caveat**（正对任务书 3.5 反面案例）；长训预算先实测再定、不按 FLOPs 猜。
7. **EMA 的实测行为（60ep 批）**：中段领先 raw +6~7 pt、末段归零（cosine lr→0 后 raw 自己收敛）→ 定位成"训练中段的稳定器"，不是终点增益；**别在报告里把它写成涨点手段**。另外它的曲线在 ep20–30 就到最终水平（想省训练时间可以提前停，但要 test 验证过才能当结论）。

### 3.4 外部方案文档不能照抄的地方（原文档已删除，结论留档）

① 编号冲突（108–113 已占用 → 全改 2xx）；② `AdamW(weight_decay=5e-4)` 太小（任务书要求 1e-2~5e-2）；③ `cudnn.benchmark=True` 不可复现 → 不进主流程；④ AMP 改数值 → 只做工程演示；⑤ "残差+3stage+加宽"打成一个 exp 违反控制变量 → 已拆成 201/202/204；⑥ 没提 TTA/EMA/集成/top-5/错误分析（任务书硬项）；⑦ 时间估计乐观 2–3×。

## 4. 不能改的约束

1. **协议**：统一 45k train / 5k val，决策只看 val；10k test 封存，只在最终 50k 重训后**测一次**。45k 与 50k 的数字不可混表。
2. **种子**：每个配置 0/42/123；**坍缩或难看的 run 一律保留并标注，绝不重跑挑好看的**；报告给 mean±std。
3. **baseline 不可改**：exp_000/101 必须与题目架构一字不差、恰好 10 epoch（它是全表的参照系）。
4. **架构必须自创**：禁止 `torchvision.models` / 照抄标准网络；`cnn_3x3` / `cnn_down` 是任务书 7a 的**起点实现**，演化方向与"为什么这样设计"的论证**由用户自己写**（评分点）。
5. **AMP 不进对照实验**；`cudnn.benchmark=True` 不进主流程。
6. **实验纪律**：贪心前向 —— 一次只改一个变量、和当前最优比；胜者成为下一轮底座。
7. **编号**：一个 `exp` = 一个配置，`_s<seed>` 区分种子；第一批占用 exp_101–109，第二批用 **exp_2xx**。
8. **GPU 纪律**：长跑前确认性能模式 + 自检；**未经用户同意不停正在跑的训练**。
9. **编辑器 clobber 坑**：程序化改完任何 `.ipynb` 后必须让用户 **Reload/Revert** 再运行，否则其保存会覆盖磁盘修改（已发生过）。配套坑：换内核或改过 import 后**必须先重跑 code #1**，否则 `NameError`。
10. **不要往实验 cell 里塞跨格依赖**：批次 cell 只允许依赖 **code #1 + code #2**；公共 helper 一律放 `task_a/src/`（已因此踩过 `NameError`）。
11. 沟通用中文；用户在本项目里**要求 agent 直接写代码 / 写 notebook**（覆盖其早先"自己写"的旧偏好，以即时指令为准）；偏好 ADHD 友好的输出（结论前置、编号步骤、动作先行）。

## 5. 剩余工作（实验已全部完成，不要再开新实验）

| 优先级 | 事项 | 说明 |
|---|---|---|
| **P0** | **写报告（英文）** | 结构照任务书 4.1，图表照 4.2，提交清单照 4.4。每一节的材料见 **§9 报告写作地图** |
| **P0** | **Part B 并入报告** | 用 `report/part_b_draft.tex`（324 行，ASCII）替换 `main.tex` 第 352 行起的骨架；`report/figs/` 里 B 的图已就绪（见 §10.3）；**B.5 需要人写**（草稿末尾有 6 条英文要点）。细节见 **§10** |
| **P1** | 提交前检查 + 打包 | notebook Execute All、清 error 输出、zip 不含 `data/` 与 `*.pt`（详见 §8） |
| P2（可选） | S3 正则线 | AdamW wd=5e-2（exp_222）/ label smoothing（223）/ Dropout（224），都跑在冠军 20.7M 上 → **~3 h/项**（60ep × 3 seeds）。**Cutout 只有 +0.6，正则化边际收益小 —— 时间不够就整段跳过，不影响已有结论** |
| P3（可选） | SGD 对照（exp_225） | 任务书因素 4 的完整对照，需重扫 lr；冠军规模下 1 seed × 2 个 lr 点 ≈ **2 h**。**报告里已有 10ep 的 SGD vs Adam（exp_101 vs 102），不做也不算缺项** |

**已完成**：S0（108/109）→ S1（200–208）→ S1b（202 复核）→ S1c（208 复核）→ S2 过夜批（211–220）→ S2c（221 锚点）→ S4a（top-5 回填）→ S4b（TTA/集成）→ S5（50k 重训）→ S6（报告图）。产物在 `task_a/runs/`、`task_a/runs_final/`、`figs/`。

**编号备忘**：60ep 锚点 = 221；S3 正则 = 222/223/224；SGD = 225；epoch 阶梯取消（60ep 曲线本身就是阶梯）。

## 6. 未解决 / 坑

| # | 事项 | 状态 |
|---|---|---|
| 1 | **top-5** | ✅ 已接入（`evaluate_topk`）；全部 74 run 回填在 `task_a/runs/topk_summary.csv`，最终模型 test top-5 = **94.36** |
| 2 | 过拟合起点 | ✅ 由 S2 的 211/213 曲线给出（ep34/ep43 见顶，见 §2.3） |
| 3 | CPU vs GPU benchmark | ✅ 已做（§2.6：7.0×） |
| 4 | notebook 从头 Execute All 能跑通 | 提交前必查（任务书标"极其重要"，本项目有 clobber 前科） |
| 5 | 慢态根因未坐实（性能模式为最可信推断） | 防线 = 自检 + 慢态警告 |
| 6 | `fit` 打印的第二组数字标签是 "test" 实为 val；`to_markdown()` 需 `pip install tabulate` | 写报告前顺手处理 |
| 7 | 坍缩精确机制只到"常数输出 + 大面积死 ReLU"层面 | 可选深度素材 |
| 7b | **CIFAR-100 的 fine 类是字母序，不是按超类分组** —— 超类映射必须从 `coarse_labels` 推，**不能 `i // 5`**（第一版写错，超类内错误率算成 2.4%，真实 36.2%）。`label_names` 的顺序只用于显示 | 已修，写报告时别再用 i//5 |
| 8 | 代码 zip 不含 `data/`、`*.pt`；notebook 无报错残留输出 | 提交前 |

## 7. 新会话必看

1. **记忆系统**：同项目新会话自动加载 `MEMORY.md` —— 先读 `dl-hw1-phase3-progress` 对齐进度；另有 `dl-hw1-collapse-seed-lottery`、`dl-hw1-gpu-slow-state`、`dl-hw1-machine-setup`、`cifar-download-fast-mirror`、`dl-hw1-taskb-layout`。**注意**：别的 agent（Codex/GPT）直接改 notebook 会把中文变成 `?`（2026-10-16 发生过）→ 用 `~/.claude/file-history` 的字节级快照恢复，见记忆 `dl-hw1-encoding-corruption-recovery`。
2. **权威数据 = `task_a/runs/`**（45k/5k 协议，主表在 `03_final_analysis.ipynb` 生成）+ **`task_a/runs_final/`**（50k 全量重训、test10k；**不要混进主表**）+ **`figs/`**（报告插图，由 `04_report_assets.ipynb` 生成）。`task_a/runs/topk_summary.csv` 是 S4a 回填的 top-1/top-5。
3. **不要动**：`.venv` 的 torch（训练中尤其）、`baseline_cnn()` 本身、`task_a/runs/exp_002_adam_45k`（坍缩证据）、LEGACY 排除清单。
4. **冠军**（截至 2026-10-17）：**`dr_wide`（20.7M）+ Adam 1e-3 + warmup/cosine + crop+flip + 60ep**。45k/5k val = 74.13（3 seeds）；**50k 重训 test = 75.01 单模型 / 77.87 集成+TTA（top-5 94.36）**。20 ep 口径最优 exp_202 = 70.40，10 ep 口径 exp_107 = 61.00 —— **三个预算是三个区块，数字不可混表**。
5. 任务书细节（验收 checklist / 规则 / 因素清单 / 报告要求）全在 `TaskA_任务书.md` —— 动手前先通读相关章节。项目协作约定在 **`agent.md`**（cell 依赖规则、`task_a/src/` 职责、UTF-8 等）；它不叫 CLAUDE.md，**不会被自动加载，要主动读**。

## 8. 提交清单（任务书 4.4）

**交付物**
- **PDF 报告单独上传 —— 用英文写**（任务书明确要求；沟通用中文不等于报告用中文）
- **代码 zip 单独上传**，里面**不要**放报告
- zip 内容：`task_a/src/` + `task_a/notebooks/`（00/01/02_main_exps_1/02_main_exps_2/03/04）+ **`task_b/npnn/` + `task_b/notebooks/` + `task_b/tests/` + `task_b/config.py` + `task_b/README.md`** + `agent.md`/`HANDOVER.md`（可选）
- **不要打包**：`data/`、任何 `*.pt`、**`task_b/runs/*/params.npz`**（`task_a/runs/` 约 2.4 GB、`task_a/runs_final/` 约 250 MB，绝大部分是权重）
- 建议打包：`task_a/runs/*/config.json` + `metrics.csv`、**`task_b/runs/*/config.json` + `metrics.csv` + `sn_weights.json`**（体积小、是"每个数字的出处"）

**提交前检查**
- [ ] 每个 notebook 从头 **Execute All** 能跑通（任务书标"极其重要"；长训练 cell 要加开关或排除）
- [ ] notebook 无 error 输出残留：`grep -c '"output_type": "error"' task_a/notebooks/*.ipynb`
- [ ] 主表由 `03_final_analysis.ipynb` 生成，数字与 `task_a/runs/` 一致
- [ ] 报告里每个数字都能在 `task_a/runs/` 或 `task_a/runs_final/` 找到出处
- [ ] 报告 §6 用**最终 test 数字**（77.87 / 94.36），不要把 val 当 test
- [ ] 图的 `dpi ≥ 150`、坐标轴有 label、图例齐全、**黑白打印也能区分线型**
- [ ] 所有超参用表格、实验对比"表格 + 图"双重呈现、图表编号并在正文引用
- [ ] 混淆矩阵按**超类**聚合观察（任务书提示：CIFAR-100 的错误大量发生在同一超类内部）

**报告里必须各自成节的素材（已就绪）**：CPU vs GPU 表（§2.6）· 数据统计与增强可视化（00_eda）· 训练曲线（task_a/runs/*/curves.png）· 消融表 + 柱状图（§2.3/2.4）· 混淆矩阵/per-class/错误样例（§2.6）· "What I learned"（§3 的七条机制）。

## 9. 报告写作地图（任务书 4.1 的结构 → 材料出处）

| 报告节 | 写什么 | 材料在哪 |
|---|---|---|
| 1. Introduction | 任务、数据、目标 | `Assignment_1.pdf`；`TaskA_任务书.md` §0 |
| 2. Environment Setup | 硬件、torch 版本、**CPU vs GPU 表**、AMP 对比 | §0 速览；§2.6（7.0×、1.89×/−518 MB） |
| 3. Dataset & Preprocessing | CIFAR-100 统计量、45k/5k/10k split、增强可视化 | `task_a/notebooks/00_eda.ipynb`；§3.4 的增强坑（别用 RandomResizedCrop 默认参数、别垂直翻转） |
| 4. Baseline | 题目架构（4,961,124 参）、10 ep、SGD 1e-2、top-1/top-5 | §2.1（exp_000/101）；`task_a/runs/exp_101_*/`；curves.png |
| 5. Experiments（20 分主战场） | 方法论 + 每个因素"动机→配置→结果→结论" | **§3 的七条机制是叙事骨架**；数字见 §2.2/2.3/2.4；图 `figs/fig_ablation_60ep.png`、`figs/fig_capacity_scaling.png`；主表由 `03_final_analysis.ipynb` 生成，top-5 在 `task_a/runs/topk_summary.csv` |
| 5.k Ablation summary | 主实验表 + 柱状图 | 同上；**三个预算区块（10/20/60 ep）不可混表**；45k 与 50k 也不可混 |
| 6. Final Model & Error Analysis | 最终配置、test 表、混淆矩阵、per-class、错误样例 | §2.5（test 表）+ §2.6；`figs/fig_confusion_fine.png`、`fig_confusion_coarse.png`、`fig_per_class.png`、`fig_error_samples.png`；`task_a/runs_final/per_class_acc.csv` |
| 7. What I Learned | 3–5 个具体、带证据的认知转变 | §3 的 1/2/3/5/7 条：坍缩与 BN、aug 的预算依赖、噪声底 1 pt、EMA 的真实作用、单进程取数 |
| 8. References | BN / ResNet / Adam / cosine / EMA / TTA 等原论文 | 用户自己补 |

**写作纪律（评分点）**
- 报告里**每个数字**都要能指到 `task_a/runs/` 或 `task_a/runs_final/` 的文件；不要凭记忆。
- 负结果照写（深度单独、depthwise、GroupNorm、Cutout 在噪声内）—— 任务书明确说负结果和失败分析是最好的证据。
- **test 数字只能出现在 §6**（最终模型）；§5 的所有对比都用 45k/5k val，并注明是验证集。
- 图的 `dpi ≥ 150`、有坐标轴 label / 图例、**黑白打印可区分**（现有图已用实线/虚线 + 不同 marker）。
- 超参用表格、实验对比"表格 + 图"双重呈现、图表编号并在正文引用。

**Part B 的报告结构不在这张表里** —— 见 §10 与 `report/part_b_draft.tex`（B.1–B.4 已写好，B.5 待写）。

---

## 10. Part B（NumPy 从零实现）— 已完成，只剩并入报告

> 2026-10-19：**Part B 的代码、验证、训练、对比全部完成**。剩下的只有「把草稿并进报告 + 写 B.5 + 编译」。

### 10.1 状态

| 项 | 状态 |
|---|---|
| 六个评分模块（1a GELU / 1b Linear / 1c CE / 1d Adam / 1e Conv2d / 1f SwitchNorm2d） | ✅ 实现 + 验收（数值梯度 + torch 对拍 + 边界形状） |
| 纯度守卫（铁律 2） | ✅ AST 静态 + 干净子进程动态 + 反向验证（fake package 必须被判失败） |
| L2 整网对拍 | ✅ 100 步 `max|Δloss| = 1.78e-15`（判据 1e-8） |
| 端到端训练（numpy） | ✅ 20 ep，train 54.4% / val 42.3%，92.4 min |
| torch 参照（同架构同预算） | ✅ train 53.6% / val 42.0%，18.6 min |
| 报告 Part B 草稿 | ✅ `report/part_b_draft.tex`（324 行、纯 ASCII、结构已查：括号/环境配平、图存在） |
| B.5「What I Learned」 | ⬜ **留给人写**（草稿末尾有 6 条英文要点，每条对应上面一处证据） |

### 10.2 关键数字（都可追溯）

| 模块 | 前向 max\|Δ\| | 反向 max rel | 数值梯度 | case 数 |
|---|---|---|---|---|
| Linear | 2.2e-16 | 3.4e-16 | 7.9e-11 | 4（含 3/4 维输入、bias=False） |
| GELU | 8.9e-16 | 3.7e-15 | 1.6e-10 | 3；与 erf 版最大差 **4.732e-4 @ x=−2.70** |
| CrossEntropyLoss | 0 | 3.5e-18 | 3.2e-9 | 3；不减 max 的版本在 1e3 logits 上得 **inf** |
| Adam | — | 2.8e-17（100 步轨迹） | 手算前 3 步逐位一致 | 存档续跑逐位一致 |
| Conv2d | 8.9e-16 | 3.8e-15 | 2.9e-11 | 8；im2col vs 朴素逐位一致；**12.8×** 加速 |
| SwitchNorm2d | 8.9e-16 | 1.2e-15（x/γ/β/λ） | 1.3e-10 | 3 + 退化测试 BN/IN/LN（≤8.9e-16） |

端到端：numpy **42.3%** vs torch **42.0%**（Δ 0.24 点）；每 epoch **255.4 s vs 49.7 s（5.1×）**。
SN 学到的分层权重：浅层 **IN 0.55**、深层 **BN 0.67 / IN 0.02** —— 论文主张的直接证据。

### 10.3 产物在哪

- **交付实现**：`task_b/npnn/`（`module/linear/activations/losses/optim/conv/norm/data`，**只许 numpy**）
- **验证工具**：`task_b/tests/`（`grad_check / test_modules / test_sn_degenerate / test_purity / l2_parity / bench_timing / train_npnn / compare_torch_train / arch`，可 import torch）
- **运行产物**：`task_b/runs/{cnn_sn, torch_ref, l2, adam, timing, smoke}/`
- **报告图**（`report/figs/`，A/B 共用，main.tex 按裸文件名引用）：
  `data_samples` · `h_sweep` · `adam_trajectory` · `l2_parity` · `l2_param_hist` · `train_curves` · `sn_weight_evolution` · `accuracy_comparison`
- **报告草稿**：`report/part_b_draft.tex`
- **notebook**：`task_b/notebooks/09–13` —— **五个都能 Execute All**；11 号「已有结果就跳过重训」，
  13 号只读 `runs/` 的 CSV/JSON

### 10.4 复现/验收命令（在 `task_b/` 下，用项目 `.venv`）

```bash
../.venv/Scripts/python.exe tests/test_purity.py          # 纯度守卫（含 --selftest）
../.venv/Scripts/python.exe tests/test_modules.py         # 六模块对拍 + SN + 数值梯度
../.venv/Scripts/python.exe tests/test_sn_degenerate.py   # SN 退化测试
../.venv/Scripts/python.exe tests/l2_parity.py            # L2 整网对拍（~12 s，会重写 runs/l2/parity.json）
../.venv/Scripts/python.exe tests/train_npnn.py --epochs 20 --out cnn_sn   # 重训 ≈90 min —— 提交前别跑
```

### 10.5 交给下一个 agent 的三件事

1. **合并草稿**：把 `report/part_b_draft.tex` 的正文粘进 `main.tex`，替换第 352 行 `\clearpage`
   之后的骨架。草稿自带 `\clearpage` 与 `\setcounter{section}{0}` / `\renewcommand{\thesection}`，
   与骨架重复，注意只留一份。
2. **写 B.5**：草稿末尾的 6 条英文要点（silent bugs / 先造尺子 / 数值稳定性 / 测量纠正直觉 /
   单一架构来源 / 诚实标注简化）扩写成第一人称段落。
3. **编译 + 提交检查**：本机**没有 LaTeX**，由用户编译；确认 `accuracy_comparison.png` 与
   `sn_weight_evolution.png` 正常显示；再跑一遍 §8 的清单。

### 10.6 Part B 的约束（改代码前必读）

- `npnn/` **只许 import numpy**（含整条 import 链）—— 改完必须跑 `test_purity.py`。
- `task_b/config.py` 不许 import torch（`npnn/data.py` 会 import 它）。
- 五个 notebook 的 **cell #1 是逐字相同的 bootstrap**，别改。
- 训练/对比脚本必须在 `import numpy` **之前**调用 `config.limit_threads()`。
- **运行期打印与异常消息一律 ASCII**（GBK 控制台会吃掉中文）；注释/docstring 可中文。
- `report/main.tex` 与 `report/figs/` 和 Part A **共用**；不要动 Part A 的数字与图。
- 已知坑：`x ** 3` 在 numpy 里走 libm `pow`（GELU 曾因此慢一倍）；架构要只从 `tests/arch.py` 生成。
