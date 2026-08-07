from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


def resolve_device(cfg):
    if hasattr(cfg, "device"):
        return cfg.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_enabled(cfg, device):
    return bool(getattr(cfg, "use_amp", False)) and device.type in {"cuda", "cpu"}


def maybe_wrap_ddp(model, cfg):
    """Best-effort DDP wrapper for notebook/runtime safety."""
    use_ddp = bool(getattr(cfg, "use_ddp", False))
    if not use_ddp:
        return model, {"enabled": False, "reason": "use_ddp=False"}

    if not torch.distributed.is_available():
        return model, {"enabled": False, "reason": "torch.distributed is unavailable"}

    if not torch.distributed.is_initialized():
        return model, {"enabled": False, "reason": "process group is not initialized"}

    if not torch.cuda.is_available():
        return model, {"enabled": False, "reason": "cuda is unavailable"}

    local_rank = int(getattr(cfg, "local_rank", 0))
    wrapped = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    return wrapped, {"enabled": True, "reason": "wrapped with DDP"}


def split_indices_for_fold(n_items, n_folds, fold_idx):
    indices = np.arange(int(n_items), dtype=np.int64)
    fold_idx = int(fold_idx)
    n_folds = int(n_folds)
    val_mask = (indices % n_folds) == fold_idx
    trn_idx = indices[~val_mask].tolist()
    val_idx = indices[val_mask].tolist()
    return trn_idx, val_idx


def build_fold_loaders(dataset, n_folds, fold_idx, batch_size=2, num_workers=0):
    trn_idx, val_idx = split_indices_for_fold(len(dataset), n_folds=n_folds, fold_idx=fold_idx)
    train_ds = Subset(dataset, trn_idx)
    val_ds = Subset(dataset, val_idx)
    train_dl = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers))
    val_dl = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))
    return train_dl, val_dl


def train_one_epoch_amp(model, loader, optimizer, loss_fn, cfg):
    device = resolve_device(cfg)
    use_amp = amp_enabled(cfg, device)
    model.train()

    scaler = None
    if use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    losses = []
    for batch in loader:
        x = batch["input"].to(device).float()
        y = batch["target"].to(device).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "n_steps": len(losses),
        "used_amp": bool(use_amp),
    }


def run_eval_simple(model, loader, loss_fn, cfg):
    device = resolve_device(cfg)
    use_amp = amp_enabled(cfg, device)
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device).float()
            y = batch["target"].to(device).float()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            losses.append(float(loss.item()))

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "n_steps": len(losses),
        "used_amp": bool(use_amp),
    }


def run_fold_loop(model_factory, dataset, cfg):
    """WBS 6-1: fold loop with AMP-ready train/eval and optional DDP gate."""
    fold_ids = list(getattr(cfg, "fold_ids", [0]))
    n_folds = int(getattr(cfg, "n_folds", max(1, len(fold_ids))))
    epochs = int(getattr(cfg, "epochs", 1))
    batch_size = int(getattr(cfg, "batch_size", 2))
    num_workers = int(getattr(cfg, "num_workers", 0))
    lr = float(getattr(cfg, "lr", 1e-3))

    device = resolve_device(cfg)
    results = {}
    for fold in fold_ids:
        model = model_factory().to(device)
        model, ddp_info = maybe_wrap_ddp(model, cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        train_dl, val_dl = build_fold_loaders(
            dataset,
            n_folds=n_folds,
            fold_idx=fold,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        fold_history = []
        for epoch in range(1, epochs + 1):
            trn = train_one_epoch_amp(model, train_dl, optimizer, loss_fn, cfg)
            val = run_eval_simple(model, val_dl, loss_fn, cfg)
            fold_history.append({"epoch": epoch, "train": trn, "val": val})

        results[int(fold)] = {
            "ddp": ddp_info,
            "history": fold_history,
            "n_train": len(train_dl.dataset),
            "n_val": len(val_dl.dataset),
        }

    return results


class _TinyFoldDataset(Dataset):
    def __init__(self, n=18, d=8, h=16, w=16):
        self.items = []
        for _ in range(int(n)):
            x = torch.randn(1, d, h, w)
            y = (torch.rand(1, d, h, w) > 0.7).float()
            self.items.append({"input": x, "target": y})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class _TinyFoldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv3d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.net(x)


def run_section_8_1_assertions():
    torch.manual_seed(123)

    ds = _TinyFoldDataset(n=18, d=8, h=16, w=16)
    cfg = SimpleNamespace(
        device=torch.device("cpu"),
        use_amp=True,
        use_ddp=True,
        local_rank=0,
        fold_ids=[0, 1, 2],
        n_folds=3,
        epochs=1,
        batch_size=3,
        num_workers=0,
        lr=1e-3,
    )

    # 6-1 validation A: fold loop returns all folds.
    out = run_fold_loop(model_factory=_TinyFoldModel, dataset=ds, cfg=cfg)
    assert set(out.keys()) == {0, 1, 2}

    # 6-1 validation B: each fold has history with finite losses.
    for fold in cfg.fold_ids:
        rec = out[int(fold)]
        assert len(rec["history"]) == cfg.epochs
        assert rec["n_train"] > 0 and rec["n_val"] > 0
        trn_loss = rec["history"][0]["train"]["loss"]
        val_loss = rec["history"][0]["val"]["loss"]
        assert np.isfinite(trn_loss)
        assert np.isfinite(val_loss)

    # 6-1 validation C: DDP gate is safe when process group is not initialized.
    ddp_reason = out[0]["ddp"]["reason"]
    assert out[0]["ddp"]["enabled"] is False
    assert isinstance(ddp_reason, str) and len(ddp_reason) > 0

    print("[6-1/folds]", sorted(out.keys()))
    print("[6-1/fold0_train_loss]", out[0]["history"][0]["train"]["loss"])
    print("[6-1/fold0_val_loss]", out[0]["history"][0]["val"]["loss"])
    print("[6-1/fold0_ddp]", out[0]["ddp"])
