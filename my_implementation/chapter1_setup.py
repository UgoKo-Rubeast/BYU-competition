from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch


@dataclass
class BaseConfig:
    seed: int = 42
    num_tomos: int = 4
    depth: int = 16
    input_size: int = 96
    batch_size: int = 1
    num_workers: int = 0
    epochs: int = 3
    lr: float = 1e-3
    label_radius: int = 1


def run_chapter1_setup(cfg=None):
    cfg = BaseConfig() if cfg is None else cfg

    assert cfg.input_size > 0, "CFG.input_size must be positive"
    assert cfg.depth > 0, "CFG.depth must be positive"
    assert cfg.num_tomos > 0, "CFG.num_tomos must be positive"

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    candidate_roots = [
        Path("../input"),
        Path("../input/competitions/byu-locating-bacterial-flagellar-motors-2025/train"),
        Path("./input"),
        Path("train_data"),
        Path("../train_data"),
    ]

    input_root = None
    for root in candidate_roots:
        if root.exists() and any(p.is_dir() and p.name.startswith("tomo_") for p in root.iterdir()):
            input_root = root
            break

    if input_root is None:
        raise FileNotFoundError(
            "tomo_* ディレクトリが見つかりません。候補: ../input, ./input, train_data, ../train_data"
        )

    assert input_root.exists(), "input_root must exist"

    print(f"[INFO] input_root = {input_root.resolve()}")
    print(f"[INFO] CFG = {cfg}")

    return {
        "CFG": cfg,
        "SEED": cfg.seed,
        "NUM_TOMOS": cfg.num_tomos,
        "DEPTH": cfg.depth,
        "IMG_SIZE": cfg.input_size,
        "BATCH_SIZE": cfg.batch_size,
        "NUM_WORKERS": cfg.num_workers,
        "EPOCHS": cfg.epochs,
        "LR": cfg.lr,
        "LABEL_RADIUS": cfg.label_radius,
        "candidate_roots": candidate_roots,
        "input_root": input_root,
    }
