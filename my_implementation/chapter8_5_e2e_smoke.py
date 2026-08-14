from types import SimpleNamespace

import numpy as np
import torch

from my_implementation.chapter8_1_amp_ddp_fold import _TinyFoldDataset, _TinyFoldModel, run_fold_loop
from my_implementation.chapter8_3_wandb_logger import get_logger
from my_implementation.chapter8_4_cfg_torch_utils import resolve_device, seed_everything, update_cfg


def _build_default_cfg():
    return SimpleNamespace(
        seed=123,
        logger="none",
        local_rank=0,
        project="byu-competition",
        run_name="wbs-6-5-e2e-smoke",
        dataset=SimpleNamespace(n=12, d=8, h=16, w=16),
        train=SimpleNamespace(
            use_amp=True,
            use_ddp=False,
            fold_ids=[0, 1],
            n_folds=2,
            epochs=1,
            batch_size=3,
            num_workers=0,
            lr=1e-3,
        ),
    )


def run_e2e_smoke(cfg=None, overrides=None, wandb_module=None):
    """WBS 6-5: end-to-end smoke run on tiny subset (train + valid)."""
    if cfg is None:
        cfg = _build_default_cfg()
    if overrides:
        cfg = update_cfg(cfg, overrides)

    seed_everything(getattr(cfg, "seed", 123), deterministic=True)
    device = resolve_device("cpu")

    train_cfg = SimpleNamespace(
        device=device,
        use_amp=bool(getattr(cfg.train, "use_amp", True)),
        use_ddp=bool(getattr(cfg.train, "use_ddp", False)),
        local_rank=int(getattr(cfg, "local_rank", 0)),
        fold_ids=list(getattr(cfg.train, "fold_ids", [0])),
        n_folds=int(getattr(cfg.train, "n_folds", 2)),
        epochs=int(getattr(cfg.train, "epochs", 1)),
        batch_size=int(getattr(cfg.train, "batch_size", 2)),
        num_workers=int(getattr(cfg.train, "num_workers", 0)),
        lr=float(getattr(cfg.train, "lr", 1e-3)),
    )

    ds = _TinyFoldDataset(
        n=int(getattr(cfg.dataset, "n", 12)),
        d=int(getattr(cfg.dataset, "d", 8)),
        h=int(getattr(cfg.dataset, "h", 16)),
        w=int(getattr(cfg.dataset, "w", 16)),
    )

    logger_cfg = SimpleNamespace(
        logger=str(getattr(cfg, "logger", "none")),
        local_rank=int(getattr(cfg, "local_rank", 0)),
        project=str(getattr(cfg, "project", "byu-competition")),
        run_name=str(getattr(cfg, "run_name", "wbs-6-5-e2e-smoke")),
        group="wbs-6-5",
        folds=train_cfg.fold_ids,
        epochs=train_cfg.epochs,
    )
    logger = get_logger(logger_cfg, wandb_module=wandb_module)

    fold_results = run_fold_loop(
        model_factory=_TinyFoldModel,
        dataset=ds,
        cfg=train_cfg,
    )

    per_fold = {}
    for fold_id, rec in sorted(fold_results.items()):
        last = rec["history"][-1]
        train_loss = float(last["train"]["loss"])
        val_loss = float(last["val"]["loss"])
        logger.log(
            {
                f"fold{fold_id}/train_loss": train_loss,
                f"fold{fold_id}/val_loss": val_loss,
                f"fold{fold_id}/n_train": int(rec["n_train"]),
                f"fold{fold_id}/n_val": int(rec["n_val"]),
            },
            commit=True,
        )
        per_fold[int(fold_id)] = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "n_train": int(rec["n_train"]),
            "n_val": int(rec["n_val"]),
            "used_amp": bool(last["train"].get("used_amp", False)),
            "ddp_enabled": bool(rec["ddp"].get("enabled", False)),
        }

    logger.finish()

    train_losses = np.array([v["train_loss"] for v in per_fold.values()], dtype=np.float64)
    val_losses = np.array([v["val_loss"] for v in per_fold.values()], dtype=np.float64)
    summary = {
        "n_folds": len(per_fold),
        "mean_train_loss": float(train_losses.mean()) if len(train_losses) > 0 else float("nan"),
        "mean_val_loss": float(val_losses.mean()) if len(val_losses) > 0 else float("nan"),
        "all_finite": bool(np.isfinite(train_losses).all() and np.isfinite(val_losses).all()),
    }

    return {
        "cfg": cfg,
        "device": str(device),
        "per_fold": per_fold,
        "summary": summary,
    }


def run_section_8_5_assertions():
    out = run_e2e_smoke(
        overrides={
            "dataset.n": "10",
            "train.fold_ids": [0, 1],
            "train.n_folds": "2",
            "train.epochs": "1",
            "train.batch_size": "2",
            "logger": "none",
        }
    )

    per_fold = out["per_fold"]
    summary = out["summary"]

    # 6-5 validation A: two folds completed.
    assert set(per_fold.keys()) == {0, 1}

    # 6-5 validation B: train/valid losses are finite and loaders are non-empty.
    for _, rec in per_fold.items():
        assert rec["n_train"] > 0 and rec["n_val"] > 0
        assert np.isfinite(rec["train_loss"])
        assert np.isfinite(rec["val_loss"])

    # 6-5 validation C: summary reports finite end-to-end run.
    assert summary["n_folds"] == 2
    assert summary["all_finite"] is True

    print("[6-5/device]", out["device"])
    print("[6-5/n_folds]", summary["n_folds"])
    print("[6-5/mean_train_loss]", summary["mean_train_loss"])
    print("[6-5/mean_val_loss]", summary["mean_val_loss"])
