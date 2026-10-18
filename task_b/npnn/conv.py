"""Conv2d（评分项 1e，10 分 ★ 最大单项）—— 参照 nn.Conv2d(..., padding=0)。

接口（任务书 §2.5）：
    Conv2d(in_channels, out_channels, kernel_size, stride=1, bias=True)
    X: (N, Cin, H, W)     W: (Cout, Cin, K, K)     b: (Cout,)
    H_out = floor((H - K) / S) + 1
    ⚠️ 题目给的 config 里没有 padding → 你的参照就是 padding=0
       （若要做成可选扩展参数，给默认值并在报告里说明这个判断）

两版都要写，不是二选一：
  · 朴素多重循环  —— oracle：先让它绝对可信，也是报告里的耗时基准
  · im2col + GEMM —— 训练路径（「向量化」要求的兑现）

✍️ TODO
- [ ] 朴素版 forward / backward
- [ ] im2col（sliding_window_view；注意开窗后的维度顺序；matmul 前 ascontiguousarray，
      不连续数组会让 BLAS 走慢路径，慢 2–3 倍）
- [ ] col2im：重叠位置必须 **累加**——np.add.at，或按窗口维度循环 +=（§2.5 头号坑）
- [ ] 两种累加写法都实现、比速度（报告里的小亮点）
- [ ] 初始化 U(-1/sqrt(Cin·K²), +1/sqrt(Cin·K²))（自己从 kaiming_uniform(a=√5) 推，别抄）
- [ ] 参数量公式写成函数 + 断言

自检（§2.5 清单，6 条一条都不能省）：
  - [ ] 多组 (N,C,H,W,K,S,bias) 前向对拍 <1e-9（至少含 K3S1 / K3S2 / K1 / K5S1 / H 不被 S 整除）
  - [ ] 反向 dx / dw / db 三个都 vs torch autograd
  - [ ] 数值梯度检验（唯一能独立于 torch 发现「重叠丢梯度」的手段）
  - [ ] im2col 版 vs 朴素版前向逐位一致
  - [ ] 形状公式手算 3 组（含 ⌊⌋ 取整边界）
  - [ ] 参数量与 sum(p.numel()) 对上
⚠️ 测试矩阵必须含 K=3,S=1：stride ≥ K 时窗口不重叠，赋值写错也「恰好正确」，会掩盖 bug。
⛔ 只允许 import numpy。
"""
