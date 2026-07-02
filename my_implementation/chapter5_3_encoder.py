from types import SimpleNamespace

import torch
import torch.nn as nn
from timm.models._manipulate import checkpoint

from my_implementation.chapter5_1_basicblock import BasicBlock
from my_implementation.chapter5_2_bottleneck import Bottleneck


def conv_out_dim(in_size, kernel_size, stride=1, padding=0, dilation=1):
    return ((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def build_linear_drop_path_rates(total_blocks, max_rate):
    if total_blocks <= 0:
        return []
    if total_blocks == 1:
        return [float(max_rate)]
    return [float(max_rate) * i / (total_blocks - 1) for i in range(total_blocks)]


def resolve_pretrained_path(backbone_name: str):
    return f"./data/model_zoo/{backbone_name}_KM_200ep.pt"


def load_weights_stub(model, wpath):
    print(f"[load_weights] called: {wpath}")


class ResnetEncoder3d(nn.Module):
    """ResNet3D encoder with DropPath schedule and feature/channel outputs."""

    BACKBONE_TABLE = {
        "r3d18": {"layers": [2, 2, 2, 2], "block": BasicBlock},
        "r3d200": {"layers": [3, 24, 36, 3], "block": Bottleneck},
    }

    def __init__(
        self,
        cfg,
        in_stride=(2, 2, 2),
        in_dilation=(1, 1, 1),
        drop_path_rate=0.2,
        inference_mode=False,
        load_weights_fn=load_weights_stub,
        use_checkpoint=False,
    ):
        super().__init__()
        self.cfg = cfg
        self.in_stride = in_stride
        self.in_dilation = in_dilation
        self.drop_path_rate = float(drop_path_rate)
        self.inference_mode = bool(inference_mode)
        self.load_weights_fn = load_weights_fn
        self.use_checkpoint = bool(use_checkpoint)
        self.pretrained_wpath = None

        backbone_name = cfg.backbone
        if backbone_name not in self.BACKBONE_TABLE:
            supported = ", ".join(self.BACKBONE_TABLE.keys())
            raise ValueError(f"ResnetEncoder3d backbone: {backbone_name} not implemented. supported=[{supported}]")

        spec = self.BACKBONE_TABLE[backbone_name]
        self.layers = spec["layers"]
        self.block = spec["block"]

        total_blocks = sum(self.layers)
        flat_rates = build_linear_drop_path_rates(total_blocks, self.drop_path_rate)
        self.block_drop_path_rates = []
        start = 0
        for n in self.layers:
            end = start + n
            self.block_drop_path_rates.append(flat_rates[start:end])
            start = end

        in_padding = tuple(d * 3 for d in in_dilation)
        self.conv1 = nn.Conv3d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7, 7),
            stride=in_stride,
            dilation=in_dilation,
            padding=in_padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.inplanes = 64
        self.layer1 = self._make_layer(self.block, planes=64, n_blocks=self.layers[0], stride=1, block_drop_path_rates=self.block_drop_path_rates[0])
        self.layer2 = self._make_layer(self.block, planes=128, n_blocks=self.layers[1], stride=2, block_drop_path_rates=self.block_drop_path_rates[1])
        self.layer3 = self._make_layer(self.block, planes=256, n_blocks=self.layers[2], stride=2, block_drop_path_rates=self.block_drop_path_rates[2])
        self.layer4 = self._make_layer(self.block, planes=512, n_blocks=self.layers[3], stride=2, block_drop_path_rates=self.block_drop_path_rates[3])

        self._maybe_load_pretrained()
        self._update_input_channels()
        self.channels = self._infer_feature_channels()

    def _maybe_load_pretrained(self):
        self.pretrained_wpath = resolve_pretrained_path(self.cfg.backbone)
        if self.inference_mode:
            print("[pretrained] skipped because inference_mode=True")
            return
        self.load_weights_fn(self, self.pretrained_wpath)

    def _update_input_channels(self):
        target_ic = int(self.cfg.in_chans)
        if self.conv1.in_channels == target_ic:
            return
        old_conv = self.conv1
        with torch.no_grad():
            w_mean = old_conv.weight.mean(dim=1, keepdim=True)
            w_new = w_mean.repeat(1, target_ic, 1, 1, 1)
        new_conv = nn.Conv3d(
            in_channels=target_ic,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            bias=False,
        )
        new_conv.weight = nn.Parameter(w_new)
        self.conv1 = new_conv

    def _infer_feature_channels(self):
        with torch.no_grad():
            was_training = self.training
            self.eval()
            x_dummy = torch.randn(
                1,
                int(self.cfg.in_chans),
                32,
                96,
                96,
                device=self.conv1.weight.device,
                dtype=self.conv1.weight.dtype,
            )
            feats = self.forward_features(x_dummy)
            channels = [f.shape[1] for f in feats]
            self.train(was_training)
        return channels

    def _checkpoint_if_enabled(self, module, x):
        if self.use_checkpoint and self.training:
            return checkpoint(module, x)
        return module(x)

    def _block_expansion(self):
        return 4 if self.block is Bottleneck else 1

    def _make_block(self, block, inplanes, planes, stride, downsample, drop_path_rate):
        if block is Bottleneck:
            return block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                expansion_factor=4,
                drop_path_rate=drop_path_rate,
            )
        return block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            drop_path_rate=drop_path_rate,
        )

    def _make_layer(self, block, planes, n_blocks, stride=1, block_drop_path_rates=None):
        if block_drop_path_rates is None:
            block_drop_path_rates = [0.0] * n_blocks
        if len(block_drop_path_rates) != n_blocks:
            raise ValueError(
                f"drop path rates length mismatch: got={len(block_drop_path_rates)} expected={n_blocks}"
            )
        expansion = self._block_expansion()
        out_channels = planes * expansion

        need_downsample = (stride != 1) or (self.inplanes != out_channels)
        downsample = None
        if need_downsample:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

        layers = [
            self._make_block(
                block=block,
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                drop_path_rate=block_drop_path_rates[0],
            )
        ]
        self.inplanes = out_channels
        for i in range(1, n_blocks):
            layers.append(
                self._make_block(
                    block=block,
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=1,
                    downsample=None,
                    drop_path_rate=block_drop_path_rates[i],
                )
            )
        return nn.Sequential(*layers)

    def forward_features(self, x):
        feats = []
        x = self._checkpoint_if_enabled(self.conv1, x)
        x = self.bn1(x)
        x = self.relu(x)
        feats.append(x)
        x = self.maxpool(x)
        x = self._checkpoint_if_enabled(self.layer1, x)
        feats.append(x)
        x = self._checkpoint_if_enabled(self.layer2, x)
        feats.append(x)
        x = self._checkpoint_if_enabled(self.layer3, x)
        feats.append(x)
        x = self._checkpoint_if_enabled(self.layer4, x)
        feats.append(x)
        return feats

    def forward(self, x):
        return self.forward_features(x)[-1]


def resolve_backbone(cfg):
    model = ResnetEncoder3d(cfg, inference_mode=True)
    return model.layers, model.block.__name__


def gather_block_drop_probs(model):
    probs = []
    for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in stage:
            dp = getattr(block, "drop_path", None)
            probs.append(float(getattr(dp, "drop_prob", 0.0)) if dp is not None else 0.0)
    return probs


class LoadRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, model, wpath):
        self.calls.append(wpath)


def run_section_5_3_assertions():
    cfg18 = SimpleNamespace(backbone="r3d18", in_chans=1)
    layers18, block18 = resolve_backbone(cfg18)
    assert layers18 == [2, 2, 2, 2] and block18 == "BasicBlock"
    cfg200 = SimpleNamespace(backbone="r3d200", in_chans=1)
    layers200, block200 = resolve_backbone(cfg200)
    assert layers200 == [3, 24, 36, 3] and block200 == "Bottleneck"

    try:
        _ = ResnetEncoder3d(SimpleNamespace(backbone="r3d999", in_chans=1), inference_mode=True)
        raise AssertionError("unsupported backbone で ValueError が必要です")
    except ValueError:
        pass

    encoder18 = ResnetEncoder3d(cfg18, drop_path_rate=0.2, inference_mode=True).eval()
    all_rates_18 = gather_block_drop_probs(encoder18)
    assert len(all_rates_18) == sum(encoder18.layers)
    assert all(all_rates_18[i] <= all_rates_18[i + 1] for i in range(len(all_rates_18) - 1))

    rec_train = LoadRecorder()
    _ = ResnetEncoder3d(cfg18, inference_mode=False, load_weights_fn=rec_train)
    assert rec_train.calls == [resolve_pretrained_path("r3d18")]
    rec_infer = LoadRecorder()
    _ = ResnetEncoder3d(cfg18, inference_mode=True, load_weights_fn=rec_infer)
    assert rec_infer.calls == []

    cfg1 = SimpleNamespace(backbone="r3d18", in_chans=1)
    m1 = ResnetEncoder3d(cfg1, inference_mode=True).eval()
    assert m1.conv1.in_channels == 1
    with torch.no_grad():
        y1 = m1(torch.randn(1, 1, 32, 96, 96))
    print(f"[3-1-11] in_chans=1 conv1.in_channels={m1.conv1.in_channels}, out={tuple(y1.shape)}")

    cfg5 = SimpleNamespace(backbone="r3d18", in_chans=5)
    m5 = ResnetEncoder3d(cfg5, inference_mode=True).eval()
    assert m5.conv1.in_channels == 5
    with torch.no_grad():
        y5 = m5(torch.randn(1, 5, 32, 96, 96))
    print(f"[3-1-11] in_chans=5 conv1.in_channels={m5.conv1.in_channels}, out={tuple(y5.shape)}")

    cfg_ckpt = SimpleNamespace(backbone="r3d18", in_chans=1)
    m_no_ckpt = ResnetEncoder3d(cfg_ckpt, inference_mode=True, use_checkpoint=False).train()
    x_no_ckpt = torch.randn(1, 1, 16, 64, 64, requires_grad=True)
    loss_no_ckpt = m_no_ckpt(x_no_ckpt).mean()
    loss_no_ckpt.backward()
    assert m_no_ckpt.conv1.weight.grad is not None

    m_ckpt = ResnetEncoder3d(cfg_ckpt, inference_mode=True, use_checkpoint=True).train()
    x_ckpt = torch.randn(1, 1, 16, 64, 64, requires_grad=True)
    loss_ckpt = m_ckpt(x_ckpt).mean()
    loss_ckpt.backward()
    assert m_ckpt.conv1.weight.grad is not None
    print("[3-1-12] checkpoint off/on: backward passed")

    cfg_feat = SimpleNamespace(backbone="r3d18", in_chans=1)
    m_feat = ResnetEncoder3d(cfg_feat, inference_mode=True).eval()
    with torch.no_grad():
        feats = m_feat.forward_features(torch.randn(1, 1, 32, 96, 96))
    assert len(feats) == 5
    feat_channels = [f.shape[1] for f in feats]
    print(f"[3-1-13] n_feats={len(feats)}, channels={feat_channels}")

    cfg_ch = SimpleNamespace(backbone="r3d18", in_chans=1)
    m_ch = ResnetEncoder3d(cfg_ch, inference_mode=True).eval()
    assert hasattr(m_ch, "channels")
    assert len(m_ch.channels) == 5
    with torch.no_grad():
        feats_ch = m_ch.forward_features(torch.randn(1, 1, 32, 96, 96))
    expected_channels = [f.shape[1] for f in feats_ch]
    assert m_ch.channels == expected_channels
    print(f"[3-1-15] channels={m_ch.channels}")
