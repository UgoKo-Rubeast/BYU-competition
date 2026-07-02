import torch
import torch.nn as nn
from timm.layers import DropPath


class Bottleneck(nn.Module):
    """1x1x1 -> 3x3x3 -> 1x1x1 bottleneck residual block."""

    def __init__(self, inplanes, planes, stride=1, downsample=False, expansion_factor=4, drop_path_rate=0.0):
        super().__init__()
        out_channels = planes * expansion_factor

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_prob=drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        if isinstance(downsample, nn.Module):
            self.downsample = downsample
        elif downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.drop_path(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


def run_section_5_2_assertions():
    x_bn = torch.randn(2, 64, 8, 32, 32)
    bottleneck_no_ds = Bottleneck(
        inplanes=64,
        planes=16,
        stride=1,
        downsample=False,
        expansion_factor=4,
        drop_path_rate=0.1,
    ).eval()
    y_bn = bottleneck_no_ds(x_bn)
    print(f"[bottleneck no_ds] input:  {tuple(x_bn.shape)}")
    print(f"[bottleneck no_ds] output: {tuple(y_bn.shape)}")
    assert y_bn.shape == x_bn.shape, "downsample=False では shape が一致する必要があります"

    x_bn_ds = torch.randn(2, 64, 8, 32, 32)
    bottleneck_ds = Bottleneck(
        inplanes=64,
        planes=32,
        stride=2,
        downsample=True,
        expansion_factor=4,
        drop_path_rate=0.1,
    ).eval()
    y_bn_ds = bottleneck_ds(x_bn_ds)
    print(f"[bottleneck ds] input:  {tuple(x_bn_ds.shape)}")
    print(f"[bottleneck ds] output: {tuple(y_bn_ds.shape)}")
    assert y_bn_ds.shape == (2, 128, 4, 16, 16), "expansion後チャネルとresidual shape を合わせる必要があります"
