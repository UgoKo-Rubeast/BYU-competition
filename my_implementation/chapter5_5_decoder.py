import torch
import torch.nn as nn
from monai.networks.blocks import UpSample

from my_implementation.chapter5_4_decoder_common import ConvBnAct3d


class DecoderBlock3d(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=nn.BatchNorm3d,
        upsample_mode="deconv",
        scale_factor=2,
    ):
        super().__init__()
        self.skip_channels = int(skip_channels)
        self.upsample = UpSample(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=in_channels,
            scale_factor=scale_factor,
            mode=upsample_mode,
        )
        self.conv1 = ConvBnAct3d(
            in_channels=in_channels + self.skip_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
        )
        self.conv2 = ConvBnAct3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if skip is None and self.skip_channels > 0:
            skip = x.new_zeros((x.size(0), self.skip_channels, *x.shape[2:]))

        if skip is not None:
            if skip.shape[2:] != x.shape[2:]:
                raise ValueError(f"skip spatial mismatch: skip={tuple(skip.shape)} x={tuple(x.shape)}")
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


def build_decoder_channel_plan(
    encoder_channels,
    decoder_channels=(256, 128, 64, 32, 16),
    skip_channels=None,
    scale_factors=(2, 2, 2, 2, 2),
):
    encoder_channels = list(encoder_channels)
    decoder_channels = list(decoder_channels)
    scale_factors = list(scale_factors)

    if len(encoder_channels) == 0:
        raise ValueError("encoder_channels must not be empty")

    n_stages = len(decoder_channels)
    if n_stages == 0:
        raise ValueError("decoder_channels must not be empty")

    if skip_channels is None:
        skip_channels = list(encoder_channels[1:]) + [0]
    else:
        skip_channels = list(skip_channels)

    if len(skip_channels) != n_stages:
        raise ValueError(
            f"skip_channels length mismatch: got={len(skip_channels)} expected={n_stages}"
        )
    if len(scale_factors) != n_stages:
        raise ValueError(
            f"scale_factors length mismatch: got={len(scale_factors)} expected={n_stages}"
        )

    in_channels = [encoder_channels[0]] + decoder_channels[:-1]
    stage_plan = []
    for idx, (ic, sc, dc, sf) in enumerate(
        zip(in_channels, skip_channels, decoder_channels, scale_factors),
        start=1,
    ):
        stage_plan.append(
            {
                "stage": idx,
                "in_channels": int(ic),
                "skip_channels": int(sc),
                "out_channels": int(dc),
                "scale_factor": int(sf),
            }
        )

    return stage_plan


class UnetDecoder3d(nn.Module):
    def __init__(
        self,
        encoder_channels,
        skip_channels=None,
        decoder_channels=(256, 128, 64, 32, 16),
        scale_factors=(2, 2, 2, 2, 2),
        norm_layer=nn.BatchNorm3d,
        upsample_mode="nontrainable",
    ):
        super().__init__()
        self.encoder_channels = tuple(encoder_channels)
        self.decoder_channels = tuple(decoder_channels)
        self.scale_factors = tuple(scale_factors)
        self.skip_channels = tuple(
            list(self.encoder_channels[1:]) + [0]
            if skip_channels is None
            else skip_channels
        )

        self.channel_plan = build_decoder_channel_plan(
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels,
            skip_channels=self.skip_channels,
            scale_factors=self.scale_factors,
        )

        self.in_channels = [s["in_channels"] for s in self.channel_plan]
        self.out_channels = [s["out_channels"] for s in self.channel_plan]

        self.blocks = nn.ModuleList(
            [
                DecoderBlock3d(
                    in_channels=stage["in_channels"],
                    skip_channels=stage["skip_channels"],
                    out_channels=stage["out_channels"],
                    norm_layer=norm_layer,
                    upsample_mode=upsample_mode,
                    scale_factor=stage["scale_factor"],
                )
                for stage in self.channel_plan
            ]
        )

    def forward(self, feats):
        if len(feats) == 0:
            raise ValueError("feats must contain at least one tensor")

        res = [feats[0]]
        skips = feats[1:]

        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            res.append(block(res[-1], skip=skip))

        return res


def extract_block_specs(decoder):
    specs = []
    for stage, block in zip(decoder.channel_plan, decoder.blocks):
        specs.append(
            {
                "conv1_in": int(block.conv1.conv.in_channels),
                "conv1_out": int(block.conv1.conv.out_channels),
                "skip_channels": int(block.skip_channels),
                "scale_factor": int(stage["scale_factor"]),
            }
        )
    return specs


class SegmentationHead3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor=(2, 2, 2),
        upsample_mode="nontrainable",
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        self.upsample = UpSample(
            spatial_dims=3,
            in_channels=out_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
            mode=upsample_mode,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


def run_section_5_5_decoder_block_assertions():
    x_main = torch.randn(2, 64, 4, 12, 12)
    skip = torch.randn(2, 16, 8, 24, 24)
    m_skip = DecoderBlock3d(
        in_channels=64,
        skip_channels=16,
        out_channels=32,
        upsample_mode="deconv",
        scale_factor=2,
    ).eval()
    with torch.no_grad():
        y_skip = m_skip(x_main, skip=skip)
    print(f"[3-2-3/with-skip] x={tuple(x_main.shape)}, skip={tuple(skip.shape)}, out={tuple(y_skip.shape)}")
    assert y_skip.shape == (2, 32, 8, 24, 24)

    m_no_skip = DecoderBlock3d(
        in_channels=64,
        skip_channels=0,
        out_channels=32,
        upsample_mode="nontrainable",
        scale_factor=2,
    ).eval()
    with torch.no_grad():
        y_no_skip = m_no_skip(x_main, skip=None)
    print(f"[3-2-3/no-skip] x={tuple(x_main.shape)}, out={tuple(y_no_skip.shape)}")
    assert y_no_skip.shape == (2, 32, 8, 24, 24)


def run_section_5_5_unet_decoder_assertions():
    plan = build_decoder_channel_plan(
        encoder_channels=[128, 64, 32, 16, 8],
        decoder_channels=(256, 128, 64, 32, 16),
        skip_channels=None,
        scale_factors=(2, 2, 2, 2, 2),
    )
    assert [s["in_channels"] for s in plan] == [128, 256, 128, 64, 32]
    assert [s["skip_channels"] for s in plan] == [64, 32, 16, 8, 0]
    assert [s["out_channels"] for s in plan] == [256, 128, 64, 32, 16]
    print("[3-2-4/default]", plan)

    try:
        _ = build_decoder_channel_plan(
            encoder_channels=[128, 64, 32, 16, 8],
            decoder_channels=(256, 128, 64),
            skip_channels=(64, 32),
            scale_factors=(2, 2, 2),
        )
        raise AssertionError("skip_channels mismatch should raise ValueError")
    except ValueError:
        pass

    decoder = UnetDecoder3d(
        encoder_channels=[128, 64, 32, 16, 8],
        decoder_channels=(256, 128, 64, 32, 16),
        skip_channels=None,
        scale_factors=(2, 2, 2, 2, 2),
        upsample_mode="nontrainable",
    )
    assert len(decoder.blocks) == 5

    specs = extract_block_specs(decoder)
    expected_conv1_in = [128 + 64, 256 + 32, 128 + 16, 64 + 8, 32 + 0]
    expected_conv1_out = [256, 128, 64, 32, 16]
    assert [s["conv1_in"] for s in specs] == expected_conv1_in
    assert [s["conv1_out"] for s in specs] == expected_conv1_out

    assert [s["scale_factor"] for s in specs] == [2, 2, 2, 2, 2]
    print("[3-2-5/blocks] n_blocks=", len(decoder.blocks))
    print("[3-2-5/specs]", specs)

    feats_dummy = [
        torch.randn(2, 128, 1, 4, 4),
        torch.randn(2, 64, 2, 8, 8),
        torch.randn(2, 32, 4, 16, 16),
        torch.randn(2, 16, 8, 32, 32),
        torch.randn(2, 8, 16, 64, 64),
    ]
    with torch.no_grad():
        decoded = decoder(feats_dummy)

    assert len(decoded) == 1 + len(decoder.blocks)
    decoded_shapes = [tuple(x.shape) for x in decoded]
    expected_shapes = [
        (2, 128, 1, 4, 4),
        (2, 256, 2, 8, 8),
        (2, 128, 4, 16, 16),
        (2, 64, 8, 32, 32),
        (2, 32, 16, 64, 64),
        (2, 16, 32, 128, 128),
    ]
    assert decoded_shapes == expected_shapes
    print("[3-2-6/decoded_shapes]", decoded_shapes)


def run_section_5_5_seg_head_assertions():
    head = SegmentationHead3d(
        in_channels=16,
        out_channels=1,
        scale_factor=(2, 2, 2),
        upsample_mode="nontrainable",
    ).eval()
    x_head = torch.randn(2, 16, 32, 128, 128)
    with torch.no_grad():
        logits = head(x_head)

    assert logits.shape == (2, 1, 64, 256, 256)
    print("[3-2-7/logits_shape]", tuple(logits.shape))

    head_no_up = SegmentationHead3d(
        in_channels=16,
        out_channels=3,
        scale_factor=(1, 1, 1),
        upsample_mode="nontrainable",
    ).eval()
    with torch.no_grad():
        logits_no_up = head_no_up(x_head)

    assert logits_no_up.shape == (2, 3, 32, 128, 128)
    print("[3-2-7/no_up_shape]", tuple(logits_no_up.shape))
