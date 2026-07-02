from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    """2-2 実装: 座標アノテーションを 3D ラベルマップへ変換して返す。"""

    def __init__(self, tomo_dirs, depth=16, img_size=96, label_radius=1, processed_root: Path | None = None):
        self.tomo_dirs = list(tomo_dirs)
        self.depth = depth
        self.img_size = img_size
        self.label_radius = label_radius
        self.processed_root = processed_root
        self.coord_meta = self._load_coord_metadata()

    def __len__(self):
        return len(self.tomo_dirs)

    def _load_slice(self, path: Path):
        img = Image.open(path).convert("L").resize((self.img_size, self.img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    def _load_coord_metadata(self):
        csv_candidates = [
            Path("./folds_all.csv"),
            Path("./data/processed/folds_all.csv"),
            Path("../input/competitions/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv"),
        ]
        csv_path = next((p for p in csv_candidates if p.exists()), None)
        if csv_path is None:
            print("[WARN] folds_all.csv が見つからないため、ラベルは全て 0 になります。")
            return {}

        df = pd.read_csv(csv_path)
        grouped = {}
        for tomo_id, g in df.groupby("tomo_id"):
            first = g.iloc[0]
            coords = []
            for z, y, x in g[["Motor axis 0", "Motor axis 1", "Motor axis 2"]].to_numpy(dtype=float):
                if z >= 0 and y >= 0 and x >= 0:
                    coords.append((z, y, x))

            grouped[tomo_id] = {
                "orig_shape": (
                    int(first["Array shape (axis 0)"]),
                    int(first["Array shape (axis 1)"]),
                    int(first["Array shape (axis 2)"]),
                ),
                "coords": coords,
            }
        return grouped

    def _scale_coord(self, coord, src_len, dst_len):
        if src_len <= 1:
            return 0
        return int(round(coord * (dst_len - 1) / (src_len - 1)))

    def _load_volume_from_images(self, tomo_dir: Path):
        slices = sorted(tomo_dir.glob("slice_*.jpg"))
        if not slices:
            slices = sorted(tomo_dir.glob("*.jpg"))
        if not slices:
            raise FileNotFoundError(f"画像スライスが見つかりません: {tomo_dir}")

        if len(slices) >= self.depth:
            sample_idx = np.linspace(0, len(slices) - 1, self.depth).astype(int)
            selected = [slices[i] for i in sample_idx]
        else:
            selected = slices + [slices[-1]] * (self.depth - len(slices))

        return np.stack([self._load_slice(p) for p in selected], axis=0)

    def _match_depth(self, vol):
        d = vol.shape[0]
        if d == self.depth:
            return vol
        if d > self.depth:
            idx = np.linspace(0, d - 1, self.depth).astype(int)
            return vol[idx]
        pad = np.repeat(vol[-1:, :, :], self.depth - d, axis=0)
        return np.concatenate([vol, pad], axis=0)

    def _load_volume(self, tomo_dir: Path):
        if self.processed_root is not None:
            npy_path = self.processed_root / f"{tomo_dir.name}.npy"
            if npy_path.exists():
                vol = np.load(npy_path).astype(np.float32)
                return self._match_depth(vol)

        return self._load_volume_from_images(tomo_dir)

    def _make_label_map(self, tomo_id: str):
        label = np.zeros((self.depth, self.img_size, self.img_size), dtype=np.float32)
        meta = self.coord_meta.get(tomo_id)
        if meta is None or len(meta["coords"]) == 0:
            return label

        src_d, src_h, src_w = meta["orig_shape"]
        r = self.label_radius
        for z0, y0, x0 in meta["coords"]:
            z = self._scale_coord(z0, src_d, self.depth)
            y = self._scale_coord(y0, src_h, self.img_size)
            x = self._scale_coord(x0, src_w, self.img_size)

            z_min, z_max = max(0, z - r), min(self.depth, z + r + 1)
            y_min, y_max = max(0, y - r), min(self.img_size, y + r + 1)
            x_min, x_max = max(0, x - r), min(self.img_size, x + r + 1)
            label[z_min:z_max, y_min:y_max, x_min:x_max] = 1.0

        return label

    def __getitem__(self, idx):
        tomo_dir = self.tomo_dirs[idx]
        vol = self._load_volume(tomo_dir)
        label = self._make_label_map(tomo_dir.name)

        x = torch.from_numpy(vol).unsqueeze(0)
        target = torch.from_numpy(label).unsqueeze(0)
        return {"input": x, "target": target, "tomo_id": tomo_dir.name}


def build_dataloader_cfg(batch_size, num_workers):
    return {
        "train": {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": num_workers,
        },
        "val": {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
        },
    }


def get_dataset(tomo_dirs, depth, img_size, label_radius, processed_root):
    return CustomDataset(
        tomo_dirs,
        depth=depth,
        img_size=img_size,
        label_radius=label_radius,
        processed_root=processed_root,
    )


def get_dataloader(dataset, mode, dataloader_cfg, shuffle=None, batch_size=None, num_workers=None):
    cfg = dataloader_cfg.get(mode, dataloader_cfg["train"])

    if shuffle is None:
        shuffle = cfg["shuffle"]
    if batch_size is None:
        batch_size = cfg["batch_size"]
    if num_workers is None:
        num_workers = cfg["num_workers"]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def run_chapter4_dataset_pipeline(
    selected_tomos,
    depth,
    img_size,
    label_radius,
    batch_size,
    num_workers,
    processed_root,
):
    dataloader_cfg = build_dataloader_cfg(batch_size=batch_size, num_workers=num_workers)

    if len(selected_tomos) >= 2:
        train_tomos = selected_tomos[:-1]
        val_tomos = selected_tomos[-1:]
    else:
        train_tomos = selected_tomos
        val_tomos = selected_tomos

    train_ds = get_dataset(train_tomos, depth, img_size, label_radius, processed_root)
    val_ds = get_dataset(val_tomos, depth, img_size, label_radius, processed_root)

    train_loader = get_dataloader(train_ds, mode="train", dataloader_cfg=dataloader_cfg)
    val_loader = get_dataloader(val_ds, mode="val", dataloader_cfg=dataloader_cfg)

    sample = next(iter(train_loader))

    assert tuple(sample["input"].shape[1:]) == (1, depth, img_size, img_size), "sample input shape mismatch"
    assert tuple(sample["target"].shape[1:]) == (1, depth, img_size, img_size), "sample target shape mismatch"

    print("[INFO] train_loader cfg =", dataloader_cfg["train"])
    print("[INFO] val_loader cfg =", dataloader_cfg["val"])
    print("[INFO] train_tomos =", [p.name for p in train_tomos])
    print("[INFO] val_tomos =", [p.name for p in val_tomos])
    print("[INFO] sample input shape =", tuple(sample["input"].shape))
    print("[INFO] sample target shape =", tuple(sample["target"].shape))
    print("[INFO] sample target positive voxels =", int(sample["target"].sum().item()))

    return {
        "DATALOADER_CFG": dataloader_cfg,
        "train_tomos": train_tomos,
        "val_tomos": val_tomos,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }
