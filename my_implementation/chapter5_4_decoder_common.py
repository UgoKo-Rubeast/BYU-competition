import torch
import torch.nn as nn


class ConvBnAct3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        norm_layer=nn.BatchNorm3d,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = norm_layer(out_channels)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def run_section_5_4_assertions():
    x_cba = torch.randn(2, 32, 8, 24, 24)
    m_cba = ConvBnAct3d(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        padding=1,
        stride=2,
    ).eval()
    with torch.no_grad():
        y_cba = m_cba(x_cba)

    print(f"[3-2-1] input shape:  {tuple(x_cba.shape)}")
    print(f"[3-2-1] output shape: {tuple(y_cba.shape)}")
    assert y_cba.shape == (2, 64, 4, 12, 12)
