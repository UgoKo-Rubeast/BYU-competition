from pathlib import Path

import numpy as np
from PIL import Image


def load_and_resize_volume(tomo_dir: Path, img_size: int):
    slice_paths = sorted(tomo_dir.glob("slice_*.jpg"))
    if not slice_paths:
        slice_paths = sorted(tomo_dir.glob("*.jpg"))
    if not slice_paths:
        raise FileNotFoundError(f"画像スライスが見つかりません: {tomo_dir}")

    volume = []
    for path in slice_paths:
        image = Image.open(path).convert("L").resize((img_size, img_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        volume.append(array)

    return np.stack(volume, axis=0)


def run_chapter3_preprocess(selected_tomos, img_size, processed_root=Path("./processed_tomos")):
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    preprocess_targets = selected_tomos

    for tomo_dir in preprocess_targets:
        volume = load_and_resize_volume(tomo_dir, img_size)
        out_path = processed_root / f"{tomo_dir.name}.npy"
        np.save(out_path, volume.astype(np.float32))

        assert volume.ndim == 3, "volume must be 3D [D, H, W]"
        assert volume.shape[1] == img_size and volume.shape[2] == img_size, "spatial size mismatch"
        assert out_path.exists(), f"preprocess output missing: {out_path}"

        print(f"[PREPROCESS] {tomo_dir.name} -> {out_path} shape={volume.shape}")

    print(f"[INFO] processed_root = {processed_root.resolve()}")
    return processed_root
