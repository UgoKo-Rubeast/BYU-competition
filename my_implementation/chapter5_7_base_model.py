from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """WBS 3-4: Base model with loss wiring and checkpoint weight loading."""

    def __init__(self, cfg: SimpleNamespace, inference_mode: bool = False):
        super().__init__()
        self.cfg = cfg
        self.inference_mode = bool(inference_mode)
        self.loss_fn = self._init_loss_fn()

    def _build_loss_kwargs(self):
        loss_cfg = getattr(self.cfg, "loss_cfg", None)
        if loss_cfg is None:
            return {}
        if isinstance(loss_cfg, dict):
            return dict(loss_cfg)
        if isinstance(loss_cfg, SimpleNamespace):
            return vars(loss_cfg)
        raise TypeError("cfg.loss_cfg must be dict or SimpleNamespace")

    def _init_loss_fn(self):
        if self.inference_mode:
            return None

        loss_type = getattr(self.cfg, "loss_type", None)
        if not loss_type:
            raise ValueError("cfg.loss_type is required when inference_mode=False")
        if "." not in loss_type:
            raise ValueError("cfg.loss_type must be a fully qualified class path")

        mname, cname = loss_type.rsplit(".", 1)
        losses = import_module(mname)
        loss_class = getattr(losses, cname)
        return loss_class(**self._build_loss_kwargs())

    def load_weights(
        self,
        weights_path=None,
        map_location="cpu",
        strict=True,
        weights_only=True,
    ):
        """Load model weights from state_dict or checkpoint dict with 'state_dict'."""
        if weights_path is None:
            weights_path = getattr(self.cfg, "weights_path", "")
        if not weights_path:
            raise ValueError("weights_path is required")

        path = Path(weights_path)
        checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        incompatible = self.load_state_dict(state_dict, strict=strict)
        return {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }


def run_section_5_7_assertions():
    class TinyBaseModel(BaseModel):
        def __init__(self, cfg, inference_mode=False):
            super().__init__(cfg=cfg, inference_mode=inference_mode)
            self.proj = nn.Conv3d(1, 1, kernel_size=1)

    # 3-4 validation A: training mode should build loss from cfg.loss_type/loss_cfg.
    cfg_train = SimpleNamespace(
        loss_type="torch.nn.BCEWithLogitsLoss",
        loss_cfg=SimpleNamespace(reduction="mean"),
    )
    model_train = TinyBaseModel(cfg_train, inference_mode=False)
    assert isinstance(model_train.loss_fn, nn.BCEWithLogitsLoss)

    # 3-4 validation B: inference mode should skip loss wiring.
    cfg_infer = SimpleNamespace(
        loss_type="torch.nn.BCEWithLogitsLoss",
        loss_cfg=SimpleNamespace(reduction="mean"),
    )
    model_infer = TinyBaseModel(cfg_infer, inference_mode=True)
    assert model_infer.loss_fn is None

    # 3-4 validation C: load_weights should restore parameters from both plain and wrapped checkpoints.
    with tempfile.TemporaryDirectory() as d:
        plain_path = Path(d) / "tiny_plain.pt"
        wrapped_path = Path(d) / "tiny_wrapped.pt"

        with torch.no_grad():
            model_train.proj.weight.fill_(0.25)
            model_train.proj.bias.fill_(0.10)

        reference_weight = model_train.proj.weight.detach().clone()
        reference_bias = model_train.proj.bias.detach().clone()

        torch.save(model_train.state_dict(), plain_path)
        torch.save({"state_dict": model_train.state_dict(), "epoch": 1}, wrapped_path)

        with torch.no_grad():
            model_train.proj.weight.zero_()
            model_train.proj.bias.zero_()

        report_plain = model_train.load_weights(plain_path, strict=True)
        assert report_plain["missing_keys"] == []
        assert report_plain["unexpected_keys"] == []
        assert torch.allclose(model_train.proj.weight, reference_weight)
        assert torch.allclose(model_train.proj.bias, reference_bias)

        with torch.no_grad():
            model_train.proj.weight.zero_()
            model_train.proj.bias.zero_()

        report_wrapped = model_train.load_weights(wrapped_path, strict=True)
        assert report_wrapped["missing_keys"] == []
        assert report_wrapped["unexpected_keys"] == []
        assert torch.allclose(model_train.proj.weight, reference_weight)
        assert torch.allclose(model_train.proj.bias, reference_bias)

    print("[3-4/loss_train]", type(model_train.loss_fn).__name__)
    print("[3-4/loss_infer]", model_infer.loss_fn)
    print("[3-4/load_plain]", report_plain)
    print("[3-4/load_wrapped]", report_wrapped)
