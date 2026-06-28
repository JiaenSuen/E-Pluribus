import torch
import torch.nn as nn
import torch.nn.functional as F


def build_EConvNeXt_Mini(num_classes, img_channels=3):
    return EConvNeXt(
        num_classes=num_classes,
        img_channels=img_channels,
        depths=[2, 2, 6, 2],
        dims=[48, 96, 192, 384],
        stage_ratio=0.5
    )


def build_EConvNeXt_Tiny(num_classes, img_channels=3):
    return EConvNeXt(
        num_classes=num_classes,
        img_channels=img_channels,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 256, 512],
        stage_ratio=0.5
    )


def build_EConvNeXt_Small(num_classes, img_channels=3):
    return EConvNeXt(
        num_classes=num_classes,
        img_channels=img_channels,
        depths=[3, 3, 15, 3],
        dims=[64, 128, 256, 512],
        stage_ratio=0.5
    )


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        act=True
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class ESEBlock(nn.Module):
    """
    Effective Squeeze-and-Excitation.
    Lightweight channel attention。
    """
    def __init__(self, channels):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.act = nn.Sigmoid()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc(scale)
        scale = self.act(scale)

        return x * scale


class EConvNeXtBlock(nn.Module):
    """
    E-ConvNeXt style block.

    Structure:
        x
        -> DWConv 7x7
        -> BN
        -> PWConv 1x1, expand 4x
        -> GELU
        -> PWConv 1x1
        -> ESE
        -> residual
    """
    def __init__(self, channels, mlp_ratio=4):
        super().__init__()

        hidden_channels = int(channels * mlp_ratio)

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=channels,
            bias=False
        )

        self.bn = nn.BatchNorm2d(channels)

        self.pwconv1 = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.act = nn.GELU()

        self.pwconv2 = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.ese = ESEBlock(channels)

        self.out_bn = nn.BatchNorm2d(channels)
 
        nn.init.constant_(self.out_bn.weight, 0.0)
        nn.init.constant_(self.out_bn.bias, 0.0)

    def forward(self, x):
        identity = x

        x = self.dwconv(x)
        x = self.bn(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.ese(x)
        x = self.out_bn(x)

        x = x + identity

        return x


class CSPConvNeXtStage(nn.Module):
    """
    CSP-style ConvNeXt stage.

    Downsample:
        Conv stride 2

    Split:
        part1 -> ConvNeXt blocks
        part2 -> shortcut

    Merge:
        concat -> 1x1 conv
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        depth,
        stage_ratio=0.5
    ):
        super().__init__()

        self.downsample = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            act=True
        )

        block_channels = int(out_channels * stage_ratio)
        shortcut_channels = out_channels - block_channels

        self.block_channels = block_channels
        self.shortcut_channels = shortcut_channels

        self.part1_conv = ConvBNAct(
            out_channels,
            block_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=True
        )

        self.part2_conv = ConvBNAct(
            out_channels,
            shortcut_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=True
        )

        self.blocks = nn.Sequential(
            *[
                EConvNeXtBlock(block_channels)
                for _ in range(depth)
            ]
        )

        self.merge_conv = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=True
        )

    def forward(self, x):
        x = self.downsample(x)

        x1 = self.part1_conv(x)
        x2 = self.part2_conv(x)

        x1 = self.blocks(x1)

        x = torch.cat([x1, x2], dim=1)
        x = self.merge_conv(x)

        return x


class SteppedStem(nn.Module):
    """
    E-ConvNeXt style stepped stem.

    224x224 -> 112x112
    後面 stage1 再降到 56x56。
    """
    def __init__(self, img_channels, out_channels):
        super().__init__()

        mid_channels = out_channels // 2

        self.stem = nn.Sequential(
            ConvBNAct(
                img_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                act=True
            ),
            ConvBNAct(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                act=True
            ),
            ConvBNAct(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                act=True
            )
        )

    def forward(self, x):
        return self.stem(x)


class EConvNeXt(nn.Module):
    """
    E-ConvNeXt-like classifier.

    forward(x):
        logits

    forward_feature_maps(x):
        returns [C2, C3, C4, C5]
        C2 stride 4
        C3 stride 8
        C4 stride 16
        C5 stride 32
    """
    def __init__(
        self,
        num_classes=1000,
        img_channels=3,
        depths=[2, 2, 6, 2],
        dims=[48, 96, 192, 384],
        stage_ratio=0.5
    ):
        super().__init__()

        self.depths = depths
        self.dims = dims
        self.out_channels = dims

        self.stem = SteppedStem(
            img_channels=img_channels,
            out_channels=dims[0]
        )

        self.stage1 = CSPConvNeXtStage(
            in_channels=dims[0],
            out_channels=dims[0],
            depth=depths[0],
            stage_ratio=stage_ratio
        )

        self.stage2 = CSPConvNeXtStage(
            in_channels=dims[0],
            out_channels=dims[1],
            depth=depths[1],
            stage_ratio=stage_ratio
        )

        self.stage3 = CSPConvNeXtStage(
            in_channels=dims[1],
            out_channels=dims[2],
            depth=depths[2],
            stage_ratio=stage_ratio
        )

        self.stage4 = CSPConvNeXtStage(
            in_channels=dims[2],
            out_channels=dims[3],
            depth=depths[3],
            stage_ratio=stage_ratio
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.BatchNorm1d(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

        self.apply(self._init_weights)

    def forward_feature_maps(self, x):
        x = self.stem(x)

        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        return [c2, c3, c4, c5]

    def forward_features(self, x):
        c2, c3, c4, c5 = self.forward_feature_maps(x)

        x = self.avgpool(c5)
        x = x.flatten(1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)

            if m.bias is not None:
                nn.init.zeros_(m.bias)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_EConvNeXt_Mini(
        num_classes=101,
        img_channels=3
    ).to(device)

    x = torch.randn(2, 3, 224, 224).to(device)

    model.eval()

    with torch.no_grad():
        y = model(x)
        features = model.forward_feature_maps(x)

    print("Model: E-ConvNeXt-Mini-like")
    print("Input:", x.shape)
    print("Output:", y.shape)
    print("Params:", count_parameters(model), "M")

    for i, f in enumerate(features, 2):
        print(f"C{i}:", f.shape)