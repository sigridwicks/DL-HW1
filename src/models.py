import torch.nn as nn

def baseline_cnn(num_classes: int = 100) -> nn.Module:
    """给定的 baseline 架构"""
    return nn.Sequential(
        nn.Conv2d(3, 256, 5, padding=2), nn.ReLU(),
        nn.Conv2d(256, 256, 5, padding=2), nn.ReLU(),
        nn.Conv2d(256, 256, 5, padding=2), nn.ReLU(),
        nn.Conv2d(256, 256, 5, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, num_classes),
    )


def baseline_cnn_bn(num_classes: int = 100) -> nn.Module:
    """baseline + BatchNorm2d after every Conv (before ReLU). Single change vs baseline_cnn."""
    return nn.Sequential(
        nn.Conv2d(3, 256, 5, padding=2), nn.BatchNorm2d(256), nn.ReLU(),
        nn.Conv2d(256, 256, 5, padding=2), nn.BatchNorm2d(256), nn.ReLU(),
        nn.Conv2d(256, 256, 5, padding=2), nn.BatchNorm2d(256), nn.ReLU(),
        nn.Conv2d(256, 256, 5, padding=2), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, num_classes),
    )


def cnn_3x3(num_classes: int = 100) -> nn.Module:
    """baseline+BN recipe, but every 5x5 Conv is replaced by two 3x3 Conv (task-doc factor 7a).

    Same receptive field (2x3x3 == 1x5x5), fewer weights (18C^2 vs 25C^2) and one extra ReLU
    per stage. Starting design for exp_106 - evolve/tune it and write the reasoning yourself.
    """
    def conv_bn_relu(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]

    layers = conv_bn_relu(3, 256) + conv_bn_relu(256, 256)      # stem: 5x5 -> 2x3x3
    for _ in range(3):                                          # the three 256->256 5x5s
        layers += conv_bn_relu(256, 256) + conv_bn_relu(256, 256)
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes)]
    return nn.Sequential(*layers)


def cnn_down(num_classes: int = 100) -> nn.Module:
    """3x3 stack + downsampling (task-doc factor 7a): thin stem, stages at 32/16/8/4 px.

    Stride-2 downsampling cuts compute ~4x per stage, which buys the extra depth at roughly
    constant cost; 2x(Conv3x3+BN+ReLU) per stage. Starting design for exp_107 - make it yours.
    """
    def stage(cin, cout, stride):
        return [nn.Conv2d(cin, cout, 3, padding=1, stride=stride), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]

    return nn.Sequential(
        *stage(3, 64, 1),       # 32x32
        *stage(64, 128, 2),     # 16x16
        *stage(128, 256, 2),    # 8x8
        *stage(256, 256, 2),    # 4x4
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, num_classes),
    )


# ---------------------------------------------------------------------------
# Batch-2 架构候选 (exp_2xx)：下面每个 wrapper 都是 cnn_down 只改一处 —— "单变量"
# 这条纪律直接写在代码里。这些是**起点实现**，按评分点要求，演化方向与"为什么这样
# 设计"的论证由你自己写。
# ---------------------------------------------------------------------------

def _norm(kind: str, channels: int, groups: int = 8) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "gn":
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unknown norm: {kind}")


def _conv_bn_relu(cin, cout, stride=1, dw_sep=False, norm="bn", groups=8):
    """一次 Conv3x3(+BN/GN)+ReLU；dw_sep=True 时换成 depthwise 3x3 + pointwise 1x1。

    写法刻意和 cnn_down 的 stage 一致（卷积默认带 bias、BN 紧跟卷积、ReLU 收尾），
    这样候选相对底座才真的只差"一个变量"。
    """
    if dw_sep:
        return [nn.Conv2d(cin, cin, 3, padding=1, stride=stride, groups=cin),   # depthwise
                _norm(norm, cin, groups), nn.ReLU(),
                nn.Conv2d(cin, cout, 1),                                        # pointwise
                _norm(norm, cout, groups), nn.ReLU()]
    return [nn.Conv2d(cin, cout, 3, padding=1, stride=stride),
            _norm(norm, cout, groups), nn.ReLU()]


class ResidualStage(nn.Module):
    """cnn_down 的 stage（两次 3x3+BN+ReLU）外面加一条恒等捷径。

    y = ReLU( bn2(conv2( ReLU(bn1(conv1(x))) )) + shortcut(x) )
    形状或通道数变化时，shortcut 用 1x1 卷积投影（stride 跟着走），否则是恒等映射。
    """

    def __init__(self, cin, cout, stride=1, norm="bn", groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, stride=stride)
        self.bn1 = _norm(norm, cout, groups)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.bn2 = _norm(norm, cout, groups)
        self.relu = nn.ReLU()
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                                          _norm(norm, cout, groups))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + (x if self.shortcut is None else self.shortcut(x)))


def cnn_down_variant(widths=(64, 128, 256, 256), strides=(1, 2, 2, 2), convs_per_stage=2,
                     residual=False, dw_sep=False, norm="bn", num_classes=100):
    """cnn_down 的可参数化版本 —— batch-2 的架构消融全部由它生成（见下面的 wrapper）。

    widths / strides  每个 stage 的输出通道数与步幅（默认 = cnn_down 的 32/16/8/4 px）
    convs_per_stage   每个 stage 里 Conv3x3 的个数（cnn_down 是 2）
    residual          每个 stage 包一条残差捷径（通道/尺寸变化处用 1x1 投影）
    dw_sep            3x3 卷积换成 depthwise + 1x1 pointwise
    norm              "bn" = BatchNorm2d，"gn" = GroupNorm(8)
    """
    assert len(widths) == len(strides), "widths 和 strides 要一一对应"
    assert not (residual and dw_sep), "残差 + depthwise 没做（要的话自己加投影分支）"
    if residual:
        assert convs_per_stage % 2 == 0, "残差块是 2 个 3x3 一组，convs_per_stage 要取偶数"

    layers, cin = [], 3
    for cout, stride in zip(widths, strides):
        if residual:
            # 每个残差块 = 2 个 3x3，所以 convs_per_stage=2 → 1 块、=4 → 2 块（深度跟着走）
            for b in range(convs_per_stage // 2):
                layers.append(ResidualStage(cin if b == 0 else cout, cout,
                                            stride if b == 0 else 1, norm))
        else:
            for j in range(convs_per_stage):
                layers += _conv_bn_relu(cin if j == 0 else cout, cout,
                                        stride if j == 0 else 1, dw_sep, norm)
        cin = cout
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, num_classes)]
    return nn.Sequential(*layers)


# 七个候选：每个都是 cnn_down + 恰好一个改动（不传参数 = 复现 cnn_down 本身）
def cnn_residual(num_classes: int = 100) -> nn.Module:
    """exp_201：+ 残差连接，形状完全不变。因素 7b。"""
    return cnn_down_variant(residual=True, num_classes=num_classes)


def cnn_wide(num_classes: int = 100) -> nn.Module:
    """exp_202：宽度 x2（64/128/256/256 -> 128/256/512/512）。因素 7f。"""
    return cnn_down_variant(widths=(128, 256, 512, 512), num_classes=num_classes)


def cnn_deep(num_classes: int = 100) -> nn.Module:
    """exp_203：深度 x2（每个 stage 的 3x3 从 2 个加到 4 个）。因素 7f。"""
    return cnn_down_variant(convs_per_stage=4, num_classes=num_classes)


def cnn_3stage(num_classes: int = 100) -> nn.Module:
    """exp_204：去掉 4px 阶段（4 -> 3 个 stage，32/16/8 px）。因素 7d。"""
    return cnn_down_variant(widths=(64, 128, 256), strides=(1, 2, 2), num_classes=num_classes)


def cnn_dw(num_classes: int = 100) -> nn.Module:
    """exp_205：3x3 -> depthwise 可分离，形状不变（参数量大幅下降）。因素 7c。"""
    return cnn_down_variant(dw_sep=True, num_classes=num_classes)


def cnn_dw_wide(num_classes: int = 100) -> nn.Module:
    """exp_206：depthwise 可分离 + 把省下的参数回投宽度（目标 ≈ 同参数量）。因素 7c。"""
    return cnn_down_variant(widths=(192, 384, 768, 768), dw_sep=True, num_classes=num_classes)


def cnn_groupnorm(num_classes: int = 100) -> nn.Module:
    """exp_207：BatchNorm -> GroupNorm(8)，通往 Task B（SwitchNorm）的桥。因素 7e。"""
    return cnn_down_variant(norm="gn", num_classes=num_classes)


def cnn_deep_residual(num_classes: int = 100) -> nn.Module:
    """exp_208：深度 x2 + 残差连接（每 stage 两个残差块）。

    单独的"深度 x2"（exp_203）和单独的"残差"（exp_201）都是负结果，这里验证经典假设
    "残差连接解锁深度"：两者组合起来能不能翻正。因素 7b + 7f。
    """
    return cnn_down_variant(convs_per_stage=4, residual=True, num_classes=num_classes)


def cnn_deep_residual_wide(num_classes: int = 100) -> nn.Module:
    """exp_217：宽度 x2 + 深度 x2 + 残差 —— 把 S1 里两个为正的旋钮叠起来。

    S1 的结论：宽度是唯一的真杠杆（+3.79），深度单独 -3.11 但"深度 + 残差"翻成 +1.45。
    这里问的是这两个赢面是否独立可加，还是互相吃掉。参数量 ≈ 2x cnn_wide。
    """
    return cnn_down_variant(widths=(128, 256, 512, 512), convs_per_stage=4,
                            residual=True, num_classes=num_classes)


# --- 容量阶梯（过夜长跑那一批）：宽度轴 1x/2x/3x，深度轴 1/2/3/4 个残差块 -------------

def cnn_deep_residual_d3(num_classes: int = 100) -> nn.Module:
    """exp_218（深 ×3）：每个 stage 3 个残差块（= 6 个 3×3），宽度保持 64/128/256/256。

    cnn_deep_residual 是 2 块/stage；这里再往上加一档，看"残差解锁深度"能走多远。
    """
    return cnn_down_variant(convs_per_stage=6, residual=True, num_classes=num_classes)


def cnn_deep_residual_d4(num_classes: int = 100) -> nn.Module:
    """exp_219（深 ×4）：每个 stage 4 个残差块（= 8 个 3×3）。

    深度轴的最后一点（再深就算力吃不消了）。三点连起来就是 depth-vs-val 曲线。
    """
    return cnn_down_variant(convs_per_stage=8, residual=True, num_classes=num_classes)


def cnn_wide3(num_classes: int = 100) -> nn.Module:
    """exp_220（宽 ×3）：宽度 192/384/768/768，深度保持 cnn_down 的 2 convs/stage。

    宽度轴的第三点（1x=2.35M / 2x=9.36M / 3x=21M），和深度轴共用同一批 60ep 预算。
    """
    return cnn_down_variant(widths=(192, 384, 768, 768), num_classes=num_classes)