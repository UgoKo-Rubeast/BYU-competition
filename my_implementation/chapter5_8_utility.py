from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch
import torch.nn as nn


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _resolve_model_class(cfg, model_registry=None):
    model_type = getattr(cfg, "model_type", None)
    if not model_type:
        raise ValueError("cfg.model_type is required")

    if model_registry is not None and model_type in model_registry:
        return model_registry[model_type]

    model_class_name = getattr(cfg, "model_class", "Net")
    module_root = getattr(cfg, "model_module_root", None)

    if "." in model_type and module_root is None:
        module = import_module(model_type)
    elif module_root:
        module = import_module(f"{module_root}.{model_type}")
    else:
        raise ValueError(
            "Cannot resolve model class. Use model_registry, set cfg.model_module_root, or pass fully-qualified cfg.model_type."
        )

    return getattr(module, model_class_name)


def _torch_load_compat(path, map_location=None, weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Older torch versions do not support weights_only.
        return torch.load(path, map_location=map_location)


def get_model(cfg, inference_mode=False, model_registry=None):
    """WBS 3-5: build model from cfg, count params, and optionally load weights."""
    model_class = _resolve_model_class(cfg, model_registry=model_registry)

    try:
        model = model_class(cfg=cfg, inference_mode=inference_mode)
    except TypeError:
        model = model_class(cfg)

    n_params = count_parameters(model)

    local_rank = int(getattr(cfg, "local_rank", 0))
    if local_rank == 0:
        print(f"Model: {getattr(cfg, 'model_type', model_class.__name__)}")
        print("n_param: {:_}".format(n_params))

    weights_path = str(getattr(cfg, "weights_path", "") or "")
    if weights_path:
        map_location = getattr(cfg, "device", "cpu")
        strict = bool(getattr(cfg, "weights_strict", True))
        checkpoint = _torch_load_compat(weights_path, map_location=map_location, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        incompatible = model.load_state_dict(state_dict, strict=strict)

        if local_rank == 0:
            print("LOADED WEIGHTS:", weights_path)
            if not strict:
                print("missing_keys:", list(incompatible.missing_keys))
                print("unexpected_keys:", list(incompatible.unexpected_keys))

    return model, n_params


def run_section_5_8_assertions():
    class TinyNet(nn.Module):
        def __init__(self, cfg, inference_mode=False):
            super().__init__()
            self.cfg = cfg
            self.inference_mode = bool(inference_mode)
            self.conv = nn.Conv3d(1, 2, kernel_size=1, bias=True)

        def forward(self, x):
            return self.conv(x)

    # 3-5 validation A: get_model should build model and return parameter count.
    cfg_build = SimpleNamespace(
        model_type="tiny",
        local_rank=0,
        weights_path="",
        device="cpu",
    )
    model_a, n_params_a = get_model(cfg_build, inference_mode=True, model_registry={"tiny": TinyNet})
    expected_params = count_parameters(model_a)
    assert isinstance(model_a, TinyNet)
    assert n_params_a == expected_params

    # 3-5 validation B/C: plain and wrapped checkpoints can be loaded.
    with tempfile.TemporaryDirectory() as d:
        plain_path = Path(d) / "tiny_plain.pt"
        wrapped_path = Path(d) / "tiny_wrapped.pt"

        with torch.no_grad():
            model_a.conv.weight.fill_(0.75)
            model_a.conv.bias.fill_(0.05)

        target_w = model_a.conv.weight.detach().clone()
        target_b = model_a.conv.bias.detach().clone()

        torch.save(model_a.state_dict(), plain_path)
        torch.save({"state_dict": model_a.state_dict(), "epoch": 9}, wrapped_path)

        cfg_plain = SimpleNamespace(
            model_type="tiny",
            local_rank=0,
            weights_path=str(plain_path),
            weights_strict=True,
            device="cpu",
        )
        model_plain, _ = get_model(cfg_plain, inference_mode=True, model_registry={"tiny": TinyNet})
        assert torch.allclose(model_plain.conv.weight, target_w)
        assert torch.allclose(model_plain.conv.bias, target_b)

        cfg_wrapped = SimpleNamespace(
            model_type="tiny",
            local_rank=0,
            weights_path=str(wrapped_path),
            weights_strict=True,
            device="cpu",
        )
        model_wrapped, _ = get_model(cfg_wrapped, inference_mode=True, model_registry={"tiny": TinyNet})
        assert torch.allclose(model_wrapped.conv.weight, target_w)
        assert torch.allclose(model_wrapped.conv.bias, target_b)

    print("[3-5/model_type]", cfg_build.model_type)
    print("[3-5/n_params]", n_params_a)
    print("[3-5/load_plain]", tuple(model_plain.conv.weight.shape))
    print("[3-5/load_wrapped]", tuple(model_wrapped.conv.weight.shape))
