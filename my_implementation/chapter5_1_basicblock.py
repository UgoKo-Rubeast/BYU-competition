import torch
import torch.nn as nn
from timm.layers import DropPath


def conv3x3x3(ic, oc, stride=1):
    return nn.Conv3d(
        in_channels=ic,
        out_channels=oc,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=False, drop_path_rate=0.0):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.drop_path = DropPath(drop_prob=drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        if isinstance(downsample, nn.Module):
            self.downsample = downsample
        elif downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    inplanes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * self.expansion),
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
        out = self.drop_path(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


def run_section_5_1_assertions():
    x_ds = torch.randn(2, 16, 8, 32, 32)
    block_ds = BasicBlock(inplanes=16, planes=32, stride=2, downsample=True, drop_path_rate=0.1)
    block_ds.eval()
    y_ds = block_ds(x_ds)
    print(f"[downsample] input shape:  {tuple(x_ds.shape)}")
    print(f"[downsample] output shape: {tuple(y_ds.shape)}")
    assert y_ds.shape == (2, 32, 4, 16, 16)
