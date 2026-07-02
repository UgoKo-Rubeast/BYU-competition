from types import SimpleNamespace

import torch
import torch.nn as nn


def build_net_class(ResnetEncoder3d, UnetDecoder3d, SegmentationHead3d):
    class Net(nn.Module):
        """UNet3D wrapper model: constructor I/O + encoder connection contract."""

        def __init__(
            self,
            cfg,
            num_classes=None,
            in_chans=None,
            deep_supervision=None,
            inference_mode=False,
        ):
            super().__init__()
            self.cfg = cfg
            self.inference_mode = bool(inference_mode)

            if not hasattr(cfg, "backbone"):
                raise ValueError("cfg.backbone is required")

            # I/O resolution rules:
            # 1) explicit argument has highest priority
            # 2) fallback to cfg field
            # 3) fallback to safe default
            self.num_classes = int(num_classes if num_classes is not None else getattr(cfg, "seg_classes", 1))
            self.in_chans = int(in_chans if in_chans is not None else getattr(cfg, "in_chans", 1))
            self.deep_supervision = bool(
                deep_supervision if deep_supervision is not None else getattr(cfg, "deep_supervision", False)
            )

            self.decoder_channels = tuple(getattr(cfg, "decoder_channels", (256, 128, 64, 32, 16)))
            self.scale_factors = tuple(getattr(cfg, "scale_factors", (2, 2, 2, 2, 2)))
            self.skip_channels = getattr(cfg, "skip_channels", None)
            self.upsample_mode = str(getattr(cfg, "upsample_mode", "nontrainable"))

            self.head_scale_factor = tuple(getattr(cfg, "head_scale_factor", (2, 2, 2)))
            self.head_upsample_mode = str(getattr(cfg, "head_upsample_mode", "nontrainable"))

            self.io_spec = {
                "required": ["cfg.backbone"],
                "optional_with_defaults": {
                    "num_classes": self.num_classes,
                    "in_chans": self.in_chans,
                    "deep_supervision": self.deep_supervision,
                    "decoder_channels": self.decoder_channels,
                    "scale_factors": self.scale_factors,
                    "skip_channels": self.skip_channels,
                    "upsample_mode": self.upsample_mode,
                    "head_scale_factor": self.head_scale_factor,
                    "head_upsample_mode": self.head_upsample_mode,
                    "inference_mode": self.inference_mode,
                },
            }

            encoder_cfg = SimpleNamespace(backbone=cfg.backbone, in_chans=self.in_chans)
            self.encoder = ResnetEncoder3d(
                cfg=encoder_cfg,
                inference_mode=self.inference_mode,
            )

            self.encoder_channels = tuple(self.encoder.channels)
            self.decoder_encoder_channels = tuple(self.encoder.channels[::-1])

            self.decoder = UnetDecoder3d(
                encoder_channels=list(self.decoder_encoder_channels),
                decoder_channels=self.decoder_channels,
                skip_channels=self.skip_channels,
                scale_factors=self.scale_factors,
                upsample_mode=self.upsample_mode,
            )

            self.seg_head = SegmentationHead3d(
                in_channels=self.decoder.decoder_channels[-1],
                out_channels=self.num_classes,
                scale_factor=self.head_scale_factor,
                upsample_mode=self.head_upsample_mode,
            )

            if self.deep_supervision:
                default_stages = (1,)
                self.deep_supervision_stages = tuple(getattr(self.cfg, "deep_supervision_stages", default_stages))

                max_valid = len(self.decoder_channels) - 1
                for stage in self.deep_supervision_stages:
                    if stage < 1 or stage > max_valid:
                        raise ValueError(
                            f"deep_supervision_stages must be in [1, {max_valid}], got={self.deep_supervision_stages}"
                        )

                self.aux_heads = nn.ModuleList(
                    [
                        SegmentationHead3d(
                            in_channels=self.decoder_channels[-1 - stage],
                            out_channels=self.num_classes,
                            scale_factor=(1, 1, 1),
                            upsample_mode="nontrainable",
                        )
                        for stage in self.deep_supervision_stages
                    ]
                )
            else:
                self.deep_supervision_stages = tuple()
                self.aux_heads = nn.ModuleList()

        def forward_encoder_features(self, x):
            """Return encoder multi-scale features in stem->stage4 order."""
            return self.encoder.forward_features(x)

        def get_encoder_feature_spec(self):
            return {
                "n_features": len(self.encoder_channels),
                "channels": self.encoder_channels,
                "decoder_input_channels": self.decoder_encoder_channels,
            }

        def forward_aux_heads(self, decoded):
            aux_logits = []
            for stage, head in zip(self.deep_supervision_stages, self.aux_heads):
                aux_logits.append(head(decoded[-1 - stage]))
            return aux_logits

        def get_deep_supervision_spec(self):
            return {
                "enabled": self.deep_supervision,
                "stages": self.deep_supervision_stages,
                "n_aux_heads": len(self.aux_heads),
                "aux_in_channels": [
                    self.decoder_channels[-1 - stage] for stage in self.deep_supervision_stages
                ],
            }

        def forward(self, batch):
            """3-3-5: basic path x -> encoder features -> decoder -> segmentation head."""
            if isinstance(batch, dict):
                x = batch["input"]
            else:
                x = batch

            if not torch.is_tensor(x):
                raise TypeError("forward input must be a torch.Tensor or a dict containing key 'input'")

            x = x.float()
            encoder_features = self.forward_encoder_features(x)
            decoder_inputs = list(encoder_features[::-1])[: len(self.decoder_channels) + 1]
            decoded = self.decoder(decoder_inputs)
            logits = self.seg_head(decoded[-1])
            return logits

    return Net


def run_section_5_6_assertions(Net):
    # 3-3-1 validation A: minimal cfg should be enough to instantiate Net
    cfg_331_min = SimpleNamespace(backbone="r3d18")
    net_331_min = Net(cfg_331_min)
    assert net_331_min.num_classes == 1
    assert net_331_min.in_chans == 1
    assert net_331_min.deep_supervision is False
    assert isinstance(net_331_min.decoder_channels, tuple) and len(net_331_min.decoder_channels) == 5

    # 3-3-1 validation B: explicit args must override cfg values
    cfg_331_custom = SimpleNamespace(
        backbone="r3d18",
        seg_classes=3,
        in_chans=2,
        deep_supervision=True,
        decoder_channels=(192, 128, 64, 32, 16),
        scale_factors=(1, 2, 2, 2, 2),
        upsample_mode="deconv",
        head_scale_factor=(1, 1, 1),
    )
    net_331_custom = Net(
        cfg_331_custom,
        num_classes=4,
        in_chans=1,
        deep_supervision=False,
    )
    assert net_331_custom.num_classes == 4
    assert net_331_custom.in_chans == 1
    assert net_331_custom.deep_supervision is False
    assert net_331_custom.upsample_mode == "deconv"
    assert tuple(net_331_custom.scale_factors) == (1, 2, 2, 2, 2)

    # 3-3-1 validation C: required field check
    try:
        _ = Net(SimpleNamespace())
        raise AssertionError("cfg.backbone missing should raise ValueError")
    except ValueError:
        pass

    print("[3-3-1/minimal]", {
        "num_classes": net_331_min.num_classes,
        "in_chans": net_331_min.in_chans,
        "deep_supervision": net_331_min.deep_supervision,
    })
    print("[3-3-1/override]", {
        "num_classes": net_331_custom.num_classes,
        "in_chans": net_331_custom.in_chans,
        "deep_supervision": net_331_custom.deep_supervision,
    })
    print("[3-3-1/io_spec_keys]", list(net_331_min.io_spec.keys()))

    # 3-3-2 validation A: encoder features are available from Net and channel spec matches
    cfg_332 = SimpleNamespace(backbone="r3d18", in_chans=1)
    net_332 = Net(cfg_332, inference_mode=True).eval()

    with torch.no_grad():
        feats_332 = net_332.forward_encoder_features(torch.randn(2, 1, 32, 96, 96))

    feat_shapes_332 = [tuple(f.shape) for f in feats_332]
    feat_channels_332 = [f.shape[1] for f in feats_332]

    assert len(feats_332) == 5
    assert tuple(feat_channels_332) == net_332.encoder_channels
    assert net_332.decoder_encoder_channels == tuple(net_332.encoder_channels[::-1])

    # 3-3-2 validation B: Net-side in_chans should be propagated to encoder
    cfg_332_ic = SimpleNamespace(backbone="r3d18", in_chans=3)
    net_332_ic = Net(cfg_332_ic, in_chans=5, inference_mode=True)
    assert net_332_ic.encoder.conv1.in_channels == 5

    print("[3-3-2/feature_shapes]", feat_shapes_332)
    print("[3-3-2/spec]", net_332.get_encoder_feature_spec())
    print("[3-3-2/in_chans]", net_332_ic.encoder.conv1.in_channels)

    # 3-3-5 validation A: basic forward path returns logits from tensor input
    cfg_335 = SimpleNamespace(
        backbone="r3d18",
        in_chans=1,
        seg_classes=2,
        head_scale_factor=(1, 1, 1),
    )
    net_335 = Net(cfg_335, inference_mode=True).eval()

    x_335 = torch.randn(2, 1, 32, 96, 96)
    with torch.no_grad():
        logits_335 = net_335(x_335)

    assert logits_335.shape == (2, 2, 32, 96, 96)

    # 3-3-5 validation B: dict input path
    with torch.no_grad():
        logits_335_dict = net_335({"input": x_335})
    assert logits_335_dict.shape == (2, 2, 32, 96, 96)

    # 3-3-5 validation C: invalid input type should fail early
    try:
        _ = net_335([1, 2, 3])
        raise AssertionError("invalid input type should raise TypeError")
    except TypeError:
        pass

    print("[3-3-5/logits_shape]", tuple(logits_335.shape))
    print("[3-3-5/logits_shape_dict]", tuple(logits_335_dict.shape))

    # 3-3-6 validation A: aux heads are built for selected decoder stages
    cfg_336 = SimpleNamespace(
        backbone="r3d18",
        in_chans=1,
        seg_classes=2,
        deep_supervision=True,
        deep_supervision_stages=(1, 2),
        head_scale_factor=(1, 1, 1),
    )
    net_336 = Net(cfg_336, inference_mode=True).eval()
    assert len(net_336.aux_heads) == 2
    assert net_336.get_deep_supervision_spec()["aux_in_channels"] == [32, 64]

    # 3-3-6 validation B: each aux head can run on mapped decoder stage output
    with torch.no_grad():
        feats_336 = net_336.forward_encoder_features(torch.randn(2, 1, 32, 96, 96))
        decoded_336 = net_336.decoder(list(feats_336[::-1])[: len(net_336.decoder_channels) + 1])
        aux_logits_336 = net_336.forward_aux_heads(decoded_336)

    assert len(aux_logits_336) == 2
    assert [x.shape[1] for x in aux_logits_336] == [2, 2]

    # 3-3-6 validation C: invalid deep supervision stage should fail early
    try:
        _ = Net(
            SimpleNamespace(
                backbone="r3d18",
                deep_supervision=True,
                deep_supervision_stages=(0,),
            )
        )
        raise AssertionError("invalid deep supervision stage should raise ValueError")
    except ValueError:
        pass

    print("[3-3-6/aux_shapes]", [tuple(x.shape) for x in aux_logits_336])
    print("[3-3-6/spec]", net_336.get_deep_supervision_spec())
