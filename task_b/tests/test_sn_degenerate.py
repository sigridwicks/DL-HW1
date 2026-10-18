"""SwitchNorm2d 的判别性测试：让 SN 退化到 BN / IN / LN（§2.6 ★ 本模块最高性价比的证据）。

把 softmax(λ) 设成 one-hot weights，SN 必须**逐位等于** PyTorch 的：
    w_BN = 1 → nn.BatchNorm2d(C, eps=1e-5)（训练模式）
    w_IN = 1 → nn.InstanceNorm2d(C, affine=True, track_running_stats=False)  ← 默认 affine=False，要显式开
    w_LN = 1 → nn.LayerNorm([C, H, W])（作用在每个样本上）

这条测试一次性验证四件事：三种统计量的维度对不对、广播对不对、
有偏/无偏方差约定对不对、ε 加在哪里（加在方差上、根号内）。
三项全过 → SN 的骨架就是对的，剩下的只是权重学习和 λ 的梯度。

✍️ TODO
- [ ] 三行退化测试，各报 max|Δ|
- [ ] 顺带打印三种统计量的 shape（§2.6 坑 1 要求的那个「看一眼」）
- [ ] （可选）顺带验证 PyTorch BN 的不对称：归一化用有偏方差、running_var 更新用无偏

运行方式：python tests/test_sn_degenerate.py（验证工具，可以 import torch）
"""
