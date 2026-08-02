from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from monai.inferers import sliding_window_inference as monai_sliding_window_inference
except Exception:
    monai_sliding_window_inference = None


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _extract_logits(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        if "main" in output:
            return output["main"]
        if "logits" in output:
            return output["logits"]
    raise TypeError("Model output must be a Tensor or a dict containing 'main' or 'logits'.")


def _get_device(cfg, model):
    if hasattr(cfg, "device"):
        return cfg.device
    try:
        return next(_unwrap_model(model).parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _to_device_batch(batch, device):
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def _scan_starts(size, roi, overlap):
    if size <= roi:
        return [0]
    stride = max(int(roi * (1.0 - overlap)), 1)
    starts = list(range(0, size - roi + 1, stride))
    if starts[-1] != size - roi:
        starts.append(size - roi)
    return starts


def _unravel_index_3d(flat_index, shape):
    d, h, w = [int(x) for x in shape]
    flat = int(flat_index)
    z = flat // (h * w)
    rem = flat % (h * w)
    y = rem // w
    x = rem % w
    return z, y, x


def _fallback_sliding_window_inference(inputs, roi_size, predictor, overlap=0.25):
    bsz, _, d, h, w = inputs.shape
    rz, ry, rx = [int(x) for x in roi_size]
    rz = min(rz, d)
    ry = min(ry, h)
    rx = min(rx, w)

    with torch.no_grad():
        probe = predictor(inputs[..., :rz, :ry, :rx])
    out_channels = int(probe.shape[1])

    out = torch.zeros((bsz, out_channels, d, h, w), device=inputs.device, dtype=probe.dtype)
    norm = torch.zeros_like(out)

    z_starts = _scan_starts(d, rz, overlap)
    y_starts = _scan_starts(h, ry, overlap)
    x_starts = _scan_starts(w, rx, overlap)

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = inputs[..., z0 : z0 + rz, y0 : y0 + ry, x0 : x0 + rx]
                pred_patch = predictor(patch)
                out[..., z0 : z0 + rz, y0 : y0 + ry, x0 : x0 + rx] += pred_patch
                norm[..., z0 : z0 + rz, y0 : y0 + ry, x0 : x0 + rx] += 1

    return out / norm.clamp_min(1)


def _sliding_window_inference(inputs, roi_size, predictor, overlap=0.25, sw_batch_size=1):
    if monai_sliding_window_inference is not None:
        return monai_sliding_window_inference(
            inputs=inputs,
            roi_size=roi_size,
            predictor=predictor,
            overlap=float(overlap),
            sw_batch_size=int(sw_batch_size),
        )

    return _fallback_sliding_window_inference(
        inputs=inputs,
        roi_size=roi_size,
        predictor=predictor,
        overlap=float(overlap),
    )


def run_eval(model, val_dl, cfg, val_metrics=None):
    """WBS 5-1: sliding-window based validation/inference loop."""
    if val_metrics is None:
        val_metrics = {"val": {}}

    device = _get_device(cfg, model)
    roi_size = tuple(getattr(cfg, "roi_size", (16, 96, 96)))
    overlap = float(getattr(cfg, "overlap", 0.25))
    sw_batch_size = int(getattr(cfg, "sw_batch_size", 1))
    use_amp = bool(getattr(cfg, "mixed_precision", False))
    disable_tqdm = bool(getattr(cfg, "disable_tqdm", True))
    use_tta = bool(getattr(cfg, "use_tta", False))
    tta_flip_dims = getattr(cfg, "tta_flip_dims", None)

    wrapped = _unwrap_model(model)
    loss_fn = getattr(wrapped, "loss_fn", None)

    model.eval()
    losses = []
    max_preds = []

    progress = tqdm(range(len(val_dl)), disable=disable_tqdm)
    val_itr = iter(val_dl)

    with torch.no_grad():
        for _ in progress:
            batch = next(val_itr)
            batch = _to_device_batch(batch, device)
            batch_input = batch["input"].float()
            target = batch.get("target", None)

            def predictor(x):
                base_model = _unwrap_model(model)
                if use_tta and hasattr(base_model, "predict") and callable(getattr(base_model, "predict")):
                    y = base_model.predict(x, use_tta=True, tta_flip_dims=tta_flip_dims)
                    return _extract_logits(y)
                return _extract_logits(model(x))

            if use_amp and device.type != "cpu":
                with torch.amp.autocast(device_type=device.type):
                    preds = _sliding_window_inference(
                        inputs=batch_input,
                        roi_size=roi_size,
                        predictor=predictor,
                        overlap=overlap,
                        sw_batch_size=sw_batch_size,
                    )
            else:
                preds = _sliding_window_inference(
                    inputs=batch_input,
                    roi_size=roi_size,
                    predictor=predictor,
                    overlap=overlap,
                    sw_batch_size=sw_batch_size,
                )

            if loss_fn is not None and torch.is_tensor(target):
                loss = loss_fn(preds, target.float())
                losses.append(float(loss.item()))

            tomo_ids = batch.get("tomo_id")
            for b in range(preds.shape[0]):
                pred_b = preds[b, 0]
                amax_idx = torch.argmax(pred_b)
                z, y, x = _unravel_index_3d(amax_idx.item(), pred_b.shape)
                prob = torch.sigmoid(pred_b[z, y, x]).item()

                if isinstance(tomo_ids, (list, tuple)):
                    tomo_id = tomo_ids[b]
                elif isinstance(tomo_ids, str):
                    tomo_id = tomo_ids
                else:
                    tomo_id = None

                max_preds.append(
                    {
                        "tomo_id": tomo_id,
                        "z": z,
                        "y": y,
                        "x": x,
                        "prob": float(prob),
                    }
                )

    val_metrics = dict(val_metrics)
    val_metrics.setdefault("val", {})
    val_metrics["val"]["n_samples"] = len(max_preds)
    val_metrics["val"]["used_tta"] = bool(use_tta)
    if losses:
        val_metrics["val"]["loss"] = float(np.mean(losses))
    if max_preds:
        val_metrics["val"]["mean_max_prob"] = float(np.mean([p["prob"] for p in max_preds]))
    val_metrics["max_preds"] = max_preds

    return val_metrics


class _TinyEvalDataset(Dataset):
    def __init__(self, n=4, d=12, h=32, w=32):
        self.items = []
        for i in range(n):
            x = torch.randn(1, d, h, w)
            y = (torch.rand(1, d, h, w) > 0.7).float()
            self.items.append({"input": x, "target": y, "tomo_id": f"tomo_{i:03d}"})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class _TinyEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=3, padding=1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.conv(x)


class _TinyEvalModelWithPredict(_TinyEvalModel):
    def __init__(self):
        super().__init__()
        self.predict_calls = 0

    def predict(self, x, use_tta=False, tta_flip_dims=None):
        self.predict_calls += 1
        base = self.forward(x)
        if not use_tta:
            return base

        if tta_flip_dims is None:
            tta_flip_dims = ((2,), (3,), (4,), (3, 4))

        logits = [base]
        for dims in tta_flip_dims:
            dims_tuple = tuple(int(d) for d in dims)
            x_flip = torch.flip(x, dims=dims_tuple)
            y_flip = self.forward(x_flip)
            y = torch.flip(y_flip, dims=dims_tuple)
            logits.append(y)
        return torch.stack(logits, dim=0).mean(dim=0)


def run_section_5_1_assertions():
    torch.manual_seed(123)

    ds = _TinyEvalDataset(n=4, d=12, h=32, w=32)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    model = _TinyEvalModel().eval()

    cfg = SimpleNamespace(
        device=torch.device("cpu"),
        roi_size=(8, 16, 16),
        overlap=0.25,
        sw_batch_size=1,
        mixed_precision=False,
        use_tta=False,
        disable_tqdm=True,
    )

    out = run_eval(model, dl, cfg, val_metrics={"val": {}})

    # 5-1 validation A: loss and sample count are produced.
    assert "val" in out
    assert out["val"]["n_samples"] == len(ds)
    assert "loss" in out["val"] and np.isfinite(out["val"]["loss"])
    assert out["val"]["used_tta"] is False

    # 5-1 validation B: one argmax prediction per sample with valid coordinates.
    assert len(out["max_preds"]) == len(ds)
    for p in out["max_preds"]:
        assert 0 <= p["z"] < 12
        assert 0 <= p["y"] < 32
        assert 0 <= p["x"] < 32
        assert np.isfinite(p["prob"])

    # 5-1 validation C: custom roi size path works and returns stable keys.
    cfg_alt = SimpleNamespace(
        device=torch.device("cpu"),
        roi_size=(12, 32, 32),
        overlap=0.5,
        sw_batch_size=1,
        mixed_precision=False,
        use_tta=False,
        disable_tqdm=True,
    )
    out_alt = run_eval(model, dl, cfg_alt, val_metrics={"val": {}})
    assert set(out_alt.keys()) == {"val", "max_preds"}
    assert "mean_max_prob" in out_alt["val"]

    # 5-4 validation A/B: Flip TTA path uses predict(...) when available.
    model_tta = _TinyEvalModelWithPredict().eval()
    cfg_tta = SimpleNamespace(
        device=torch.device("cpu"),
        roi_size=(8, 16, 16),
        overlap=0.25,
        sw_batch_size=1,
        mixed_precision=False,
        use_tta=True,
        tta_flip_dims=((2,), (3,), (4,)),
        disable_tqdm=True,
    )
    out_tta = run_eval(model_tta, dl, cfg_tta, val_metrics={"val": {}})
    assert out_tta["val"]["used_tta"] is True
    assert model_tta.predict_calls > 0
    assert len(out_tta["max_preds"]) == len(ds)

    print("[5-1/n_samples]", out["val"]["n_samples"])
    print("[5-1/loss]", out["val"]["loss"])
    print("[5-1/mean_max_prob]", out["val"]["mean_max_prob"])
    print("[5-4/used_tta]", out_tta["val"]["used_tta"])
    print("[5-4/predict_calls]", model_tta.predict_calls)
