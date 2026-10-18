import torch.nn as nn

def baseline_cnn(num_classes: int = 100) -> nn.Module:
    """Baseline architecture from the assignment."""
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
    """Baseline + BatchNorm2d after every Conv (before ReLU). Single change vs baseline_cnn."""
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
    """Replace every 5x5 Conv with two 3x3 Convs (task-doc factor 7a).

    Same receptive field (2x3x3 == 1x5x5), fewer weights (18C^2 vs 25C^2),
    and one extra ReLU per stage. Starting design for exp_106.
    """
    def conv_bn_relu(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]

    layers = conv_bn_relu(3, 256) + conv_bn_relu(256, 256)      # stem: 5x5 -> 2x3x3
    for _ in range(3):                                          # the three 256->256 5x5s
        layers += conv_bn_relu(256, 256) + conv_bn_relu(256, 256)
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes)]
    return nn.Sequential(*layers)


def cnn_down(num_classes: int = 100) -> nn.Module:
    """3x3 stack with stride-2 downsampling (task-doc factor 7a).

    A thin stem followed by stages at 32/16/8/4 px. Downsampling cuts compute
    roughly 4x per stage, which buys extra depth at similar cost.
    Each stage uses two (Conv3x3 + BN + ReLU) blocks. Starting design for exp_107.
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
# Batch-2 architecture candidates (exp_2xx): each wrapper changes exactly one
# thing in cnn_down. These are starting points; the design rationale is part
# of the report.
# ---------------------------------------------------------------------------

def _norm(kind: str, channels: int, groups: int = 8) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "gn":
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unknown norm: {kind}")


def _conv_bn_relu(cin, cout, stride=1, dw_sep=False, norm="bn", groups=8):
    """One Conv3x3 (+BN/GN) + ReLU. With dw_sep=True, use depthwise 3x3 + pointwise 1x1.

    The layout matches cnn_down's stage so each candidate differs by one variable only.
    """
    if dw_sep:
        return [nn.Conv2d(cin, cin, 3, padding=1, stride=stride, groups=cin),   # depthwise
                _norm(norm, cin, groups), nn.ReLU(),
                nn.Conv2d(cin, cout, 1),                                        # pointwise
                _norm(norm, cout, groups), nn.ReLU()]
    return [nn.Conv2d(cin, cout, 3, padding=1, stride=stride),
            _norm(norm, cout, groups), nn.ReLU()]


class ResidualStage(nn.Module):
    """A cnn_down stage (two 3x3+BN+ReLU blocks) with an identity shortcut.

    y = ReLU(bn2(conv2(ReLU(bn1(conv1(x))))) + shortcut(x)).
    A 1x1 projection is used when the shape or channel count changes.
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
    """Parameterized cnn_down used by all batch-2 architecture ablations.

    widths / strides   output channels and stride per stage (defaults match cnn_down)
    convs_per_stage    number of Conv3x3 blocks per stage (cnn_down uses 2)
    residual           wrap each stage in a residual block (1x1 projection on shape change)
    dw_sep             replace 3x3 convs with depthwise + 1x1 pointwise
    norm               "bn" = BatchNorm2d, "gn" = GroupNorm(8)
    """
    assert len(widths) == len(strides), "widths and strides must have the same length"
    assert not (residual and dw_sep), "residual + depthwise is not implemented"
    if residual:
        assert convs_per_stage % 2 == 0, "a residual block uses two 3x3 convs; convs_per_stage must be even"

    layers, cin = [], 3
    for cout, stride in zip(widths, strides):
        if residual:
            # One residual block = two 3x3 convs: convs_per_stage=2 -> 1 block, 4 -> 2 blocks.
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


# Seven candidates: each is cnn_down plus exactly one change (no arguments = cnn_down itself).
def cnn_residual(num_classes: int = 100) -> nn.Module:
    """exp_201: add residual connections, no shape change. Factor 7b."""
    return cnn_down_variant(residual=True, num_classes=num_classes)


def cnn_wide(num_classes: int = 100) -> nn.Module:
    """exp_202: width x2 (64/128/256/256 -> 128/256/512/512). Factor 7f."""
    return cnn_down_variant(widths=(128, 256, 512, 512), num_classes=num_classes)


def cnn_deep(num_classes: int = 100) -> nn.Module:
    """exp_203: depth x2 (two -> four 3x3 convs per stage). Factor 7f."""
    return cnn_down_variant(convs_per_stage=4, num_classes=num_classes)


def cnn_3stage(num_classes: int = 100) -> nn.Module:
    """exp_204: drop the 4px stage (four -> three stages, 32/16/8 px). Factor 7d."""
    return cnn_down_variant(widths=(64, 128, 256), strides=(1, 2, 2), num_classes=num_classes)


def cnn_dw(num_classes: int = 100) -> nn.Module:
    """exp_205: replace 3x3 convs with depthwise-separable convs. Factor 7c."""
    return cnn_down_variant(dw_sep=True, num_classes=num_classes)


def cnn_dw_wide(num_classes: int = 100) -> nn.Module:
    """exp_206: depthwise-separable convs plus wider channels (roughly constant params). Factor 7c."""
    return cnn_down_variant(widths=(192, 384, 768, 768), dw_sep=True, num_classes=num_classes)


def cnn_groupnorm(num_classes: int = 100) -> nn.Module:
    """exp_207: BatchNorm -> GroupNorm(8), a bridge to Task B (SwitchNorm). Factor 7e."""
    return cnn_down_variant(norm="gn", num_classes=num_classes)


def cnn_deep_residual(num_classes: int = 100) -> nn.Module:
    """exp_208: depth x2 + residual connections (two residual blocks per stage).

    Depth alone (exp_203) and residuals alone (exp_201) were both negative.
    This tests the classic hypothesis that residuals unlock depth. Factors 7b + 7f.
    """
    return cnn_down_variant(convs_per_stage=4, residual=True, num_classes=num_classes)


def cnn_deep_residual_wide(num_classes: int = 100) -> nn.Module:
    """exp_217: width x2 + depth x2 + residuals, combining the two positive knobs from S1.

    S1 found width to be the only strong lever (+3.79); depth alone was -3.11,
    but depth + residuals turned into +1.45. This asks whether the two wins add.
    """
    return cnn_down_variant(widths=(128, 256, 512, 512), convs_per_stage=4,
                            residual=True, num_classes=num_classes)


# --- Capacity ladders: width 1x/2x/3x, depth 1/2/3/4 residual blocks per stage ---

def cnn_deep_residual_d3(num_classes: int = 100) -> nn.Module:
    """exp_218 (depth x3): three residual blocks per stage (six 3x3 convs)."""
    return cnn_down_variant(convs_per_stage=6, residual=True, num_classes=num_classes)


def cnn_deep_residual_d4(num_classes: int = 100) -> nn.Module:
    """exp_219 (depth x4): four residual blocks per stage (eight 3x3 convs)."""
    return cnn_down_variant(convs_per_stage=8, residual=True, num_classes=num_classes)


def cnn_wide3(num_classes: int = 100) -> nn.Module:
    """exp_220 (width x3): 192/384/768/768 channels, keeping cnn_down's depth."""
    return cnn_down_variant(widths=(192, 384, 768, 768), num_classes=num_classes)
