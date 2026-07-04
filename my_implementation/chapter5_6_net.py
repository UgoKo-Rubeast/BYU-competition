from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                # 3-3-7: keep deterministic stage order for loss calculation.
                self.deep_supervision_stages = tuple(
                    sorted(set(getattr(self.cfg, "deep_supervision_stages", default_stages)))
                )

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

        def forward_aux_heads(self, decoded, output_size=None):
            aux_logits = []
            for stage, head in zip(self.deep_supervision_stages, self.aux_heads):
                aux = head(decoded[-1 - stage])
                # 3-3-7: unify aux-logits spatial size to make loss computation straightforward.
                if output_size is not None and aux.shape[2:] != output_size:
                    aux = F.interpolate(aux, size=output_size, mode="trilinear", align_corners=False)
                aux_logits.append(aux)
            return aux_logits

        def build_deep_supervision_outputs(self, decoded, main_logits):
            """Return [main, aux1, aux2, ...] with unified resolution and deterministic order."""
            aux_logits = self.forward_aux_heads(decoded, output_size=main_logits.shape[2:])
            return [main_logits] + aux_logits

        def _resize_logits_if_needed(self, logits, output_size):
            if output_size is None or logits.shape[2:] == tuple(output_size):
                return logits
            return F.interpolate(logits, size=tuple(output_size), mode="trilinear", align_corners=False)

        def _align_feature_size(self, feat, size):
            if feat.shape[2:] == tuple(size):
                return feat
            return F.interpolate(feat, size=tuple(size), mode="trilinear", align_corners=False)

        def _prepare_decoder_inputs(self, encoder_features):
            """3-3-9: align skip feature shapes for odd-size inputs before decoder forward."""
            decoder_inputs = list(encoder_features[::-1])[: len(self.decoder_channels) + 1]
            if len(decoder_inputs) <= 1:
                return decoder_inputs

            cur_size = tuple(decoder_inputs[0].shape[2:])
            aligned = [decoder_inputs[0]]
            for i in range(1, len(decoder_inputs)):
                sf = int(self.scale_factors[i - 1]) if i - 1 < len(self.scale_factors) else 2
                target_size = tuple(max(1, s * sf) for s in cur_size)
                aligned_skip = self._align_feature_size(decoder_inputs[i], target_size)
                aligned.append(aligned_skip)
                cur_size = target_size
            return aligned

        def get_deep_supervision_spec(self):
            return {
                "enabled": self.deep_supervision,
                "stages": self.deep_supervision_stages,
                "n_aux_heads": len(self.aux_heads),
                "aux_in_channels": [
                    self.decoder_channels[-1 - stage] for stage in self.deep_supervision_stages
                ],
            }

        def _format_outputs(self, main_logits, decoded, return_dict=None, output_size=None):
            # 3-3-8: default behavior is train=dict(deep supervision info), eval=tensor(main only).
            if return_dict is None:
                return_dict = bool(self.training)

            if output_size is not None:
                main_logits = self._resize_logits_if_needed(main_logits, output_size)

            if not return_dict:
                return main_logits

            outputs = {"main": main_logits}
            if self.deep_supervision:
                outputs["aux"] = self.forward_aux_heads(decoded, output_size=main_logits.shape[2:])
                outputs["all"] = [main_logits] + outputs["aux"]
            else:
                outputs["aux"] = []
                outputs["all"] = [main_logits]
            return outputs

        def forward(self, batch, return_dict=None):
            """3-3-5/3-3-8: basic path + train/infer return branching."""
            target = None
            if isinstance(batch, dict):
                x = batch["input"]
                target = batch.get("target")
            else:
                x = batch

            if not torch.is_tensor(x):
                raise TypeError("forward input must be a torch.Tensor or a dict containing key 'input'")

            x = x.float()
            encoder_features = self.forward_encoder_features(x)
            decoder_inputs = self._prepare_decoder_inputs(encoder_features)
            decoded = self.decoder(decoder_inputs)
            main_logits = self.seg_head(decoded[-1])

            # 3-3-9: enforce final shape consistency for odd sizes and loss-target compatibility.
            output_size = target.shape[2:] if torch.is_tensor(target) else x.shape[2:]
            return self._format_outputs(
                main_logits,
                decoded,
                return_dict=return_dict,
                output_size=output_size,
            )

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

    # 3-3-7 validation A: deep supervision stage order is normalized
    cfg_337 = SimpleNamespace(
        backbone="r3d18",
        in_chans=1,
        seg_classes=2,
        deep_supervision=True,
        deep_supervision_stages=(2, 1),
        head_scale_factor=(1, 1, 1),
    )
    net_337 = Net(cfg_337, inference_mode=True).eval()
    assert net_337.deep_supervision_stages == (1, 2)

    # 3-3-7 validation B: main + aux outputs are returned in unified resolution and fixed order
    with torch.no_grad():
        x_337 = torch.randn(2, 1, 32, 96, 96)
        feats_337 = net_337.forward_encoder_features(x_337)
        decoded_337 = net_337.decoder(list(feats_337[::-1])[: len(net_337.decoder_channels) + 1])
        main_337 = net_337.seg_head(decoded_337[-1])
        ds_outputs_337 = net_337.build_deep_supervision_outputs(decoded_337, main_337)

    assert len(ds_outputs_337) == 1 + len(net_337.aux_heads)
    assert ds_outputs_337[0].shape == main_337.shape
    assert all(x.shape[2:] == main_337.shape[2:] for x in ds_outputs_337[1:])
    assert [x.shape[1] for x in ds_outputs_337] == [net_337.num_classes] * len(ds_outputs_337)

    print("[3-3-7/stages]", net_337.deep_supervision_stages)
    print("[3-3-7/ds_shapes]", [tuple(x.shape) for x in ds_outputs_337])

    # 3-3-8 validation A: training mode returns dict by default
    net_338_train = Net(cfg_337, inference_mode=True).train()
    out_train = net_338_train(torch.randn(2, 1, 32, 96, 96))
    assert isinstance(out_train, dict)
    assert set(out_train.keys()) == {"main", "aux", "all"}
    assert out_train["all"][0].shape == out_train["main"].shape
    assert len(out_train["aux"]) == len(net_338_train.aux_heads)

    # 3-3-8 validation B: eval mode returns main tensor by default
    net_338_eval = Net(cfg_337, inference_mode=True).eval()
    with torch.no_grad():
        out_eval = net_338_eval(torch.randn(2, 1, 32, 96, 96))
    assert torch.is_tensor(out_eval)
    assert out_eval.shape[1] == net_338_eval.num_classes

    # 3-3-8 validation C: explicit return_dict flag overrides default behavior
    with torch.no_grad():
        out_eval_dict = net_338_eval(torch.randn(2, 1, 32, 96, 96), return_dict=True)
    assert isinstance(out_eval_dict, dict)
    out_train_tensor = net_338_train(torch.randn(2, 1, 32, 96, 96), return_dict=False)
    assert torch.is_tensor(out_train_tensor)

    print("[3-3-8/train_keys]", list(out_train.keys()))
    print("[3-3-8/eval_main_shape]", tuple(out_eval.shape))
    print("[3-3-8/override_eval_dict_keys]", list(out_eval_dict.keys()))

    # 3-3-9 validation A: odd-size input should return output aligned to input size
    cfg_339_odd = SimpleNamespace(
        backbone="r3d18",
        in_chans=1,
        seg_classes=2,
        deep_supervision=False,
    )
    net_339_odd = Net(cfg_339_odd, inference_mode=True).eval()
    x_odd = torch.randn(2, 1, 17, 95, 97)
    with torch.no_grad():
        logits_odd = net_339_odd(x_odd)
    assert logits_odd.shape[2:] == x_odd.shape[2:]

    # 3-3-9 validation B: dict input with target should align output size to target for loss compatibility
    cfg_339_ds = SimpleNamespace(
        backbone="r3d18",
        in_chans=1,
        seg_classes=2,
        deep_supervision=True,
        deep_supervision_stages=(1, 2),
    )
    net_339_ds = Net(cfg_339_ds, inference_mode=True).train()
    x_339 = torch.randn(2, 1, 19, 93, 91)
    target_339 = torch.randn(2, 1, 16, 96, 96)
    out_339 = net_339_ds({"input": x_339, "target": target_339})
    assert isinstance(out_339, dict)
    assert out_339["main"].shape[2:] == target_339.shape[2:]
    assert all(aux.shape[2:] == target_339.shape[2:] for aux in out_339["aux"])

    print("[3-3-9/odd_input_out_shape]", tuple(logits_odd.shape))
    print("[3-3-9/target_aligned_main_shape]", tuple(out_339["main"].shape))
