import random
from types import SimpleNamespace

import numpy as np
import torch


def _coerce_value(raw_value, current_value):
    if isinstance(current_value, bool):
        return str(raw_value).lower() in {"1", "true", "yes", "on"}
    if current_value is None:
        return raw_value
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, str):
        return str(raw_value)
    return raw_value


def _resolve_parent_and_key(cfg, dotted_key):
    parts = str(dotted_key).split(".")
    parent = cfg
    for part in parts[:-1]:
        if isinstance(parent, dict):
            if part not in parent:
                raise KeyError(f"missing config path: {dotted_key}")
            parent = parent[part]
        else:
            if not hasattr(parent, part):
                raise KeyError(f"missing config path: {dotted_key}")
            parent = getattr(parent, part)
    return parent, parts[-1]


def _get_value(container, key):
    if isinstance(container, dict):
        if key not in container:
            raise KeyError(f"unknown config key: {key}")
        return container[key]
    if not hasattr(container, key):
        raise KeyError(f"unknown config key: {key}")
    return getattr(container, key)


def _set_value(container, key, value):
    if isinstance(container, dict):
        container[key] = value
    else:
        setattr(container, key, value)


def update_cfg(cfg, overrides, log=False):
    """WBS 6-4: update cfg fields from override dict with type coercion."""
    for key, raw_value in dict(overrides).items():
        parent, leaf_key = _resolve_parent_and_key(cfg, key)
        before = _get_value(parent, leaf_key)
        after = _coerce_value(raw_value, before)
        _set_value(parent, leaf_key, after)

        if log:
            print(f"[6-4/update_cfg] {key}: {before} -> {after}")
    return cfg


def seed_everything(seed, deterministic=False):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def resolve_device(preferred=None):
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [to_device(v, device) for v in batch]
        return type(batch)(moved)
    return batch


def state_dict_to_cpu(state_dict):
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in state_dict.items()}


def count_trainable_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def run_section_8_4_assertions():
    # 6-4 validation A: cfg override with nested keys and type coercion.
    cfg = SimpleNamespace(
        lr=1e-3,
        epochs=3,
        use_amp=False,
        model=SimpleNamespace(depth=16, name="tiny"),
    )
    update_cfg(
        cfg,
        {
            "lr": "0.0025",
            "epochs": "7",
            "use_amp": "True",
            "model.depth": "32",
            "model.name": "mini",
        },
    )
    assert abs(cfg.lr - 0.0025) < 1e-12
    assert cfg.epochs == 7
    assert cfg.use_amp is True
    assert cfg.model.depth == 32
    assert cfg.model.name == "mini"

    # 6-4 validation B: seed reproducibility.
    seed_everything(123, deterministic=True)
    a1 = torch.rand(4)
    seed_everything(123, deterministic=True)
    a2 = torch.rand(4)
    assert torch.allclose(a1, a2)

    # 6-4 validation C: recursive device move on nested payload.
    payload = {
        "x": torch.randn(2, 3),
        "nested": [torch.randn(1), {"y": torch.randn(1)}],
        "meta": "keep",
    }
    device = resolve_device("cpu")
    moved = to_device(payload, device)
    assert moved["x"].device.type == "cpu"
    assert moved["nested"][0].device.type == "cpu"
    assert moved["nested"][1]["y"].device.type == "cpu"
    assert moved["meta"] == "keep"

    # 6-4 validation D: parameter counting and CPU state dict export.
    model = torch.nn.Sequential(
        torch.nn.Conv3d(1, 4, kernel_size=3, padding=1, bias=False),
        torch.nn.ReLU(),
        torch.nn.Conv3d(4, 1, kernel_size=1, bias=True),
    )
    n_params = count_trainable_params(model)
    assert n_params > 0
    cpu_state = state_dict_to_cpu(model.state_dict())
    assert all((not torch.is_tensor(v)) or (v.device.type == "cpu") for v in cpu_state.values())

    print("[6-4/cfg_lr]", cfg.lr)
    print("[6-4/cfg_epochs]", cfg.epochs)
    print("[6-4/cfg_use_amp]", cfg.use_amp)
    print("[6-4/seed_repro]", bool(torch.allclose(a1, a2)))
    print("[6-4/n_params]", n_params)
