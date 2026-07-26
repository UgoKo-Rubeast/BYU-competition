from types import SimpleNamespace

import torch
import torch.nn.functional as F


def nms_2d(scores: torch.Tensor, nms_radius: int):
    """Apply local-maximum suppression on 2D score maps."""
    if nms_radius < 0:
        raise ValueError("nms_radius must be >= 0")

    def max_pool(x):
        return F.max_pool2d(
            x,
            kernel_size=nms_radius * 2 + 1,
            stride=1,
            padding=nms_radius,
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    return torch.where(max_mask, scores, zeros)


def nms_3d(scores: torch.Tensor, nms_radius: int):
    """Apply local-maximum suppression on 3D score volumes."""
    if nms_radius < 0:
        raise ValueError("nms_radius must be >= 0")

    def max_pool(x):
        return F.max_pool3d(
            x,
            kernel_size=nms_radius * 2 + 1,
            stride=1,
            padding=nms_radius,
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    return torch.where(max_mask, scores, zeros)


def extract_topk_after_nms_3d(
    scores_3d: torch.Tensor,
    nms_radius: int = 2,
    topk: int = 1,
    score_threshold: float = 0.0,
):
    """
    Postprocess one 3D score volume into top-k peak coordinates.

    Args:
        scores_3d: [D, H, W] tensor.
    Returns:
        list of dicts: [{"z": int, "y": int, "x": int, "score": float}, ...]
    """
    if scores_3d.ndim != 3:
        raise ValueError("scores_3d must be 3D [D, H, W]")
    if topk <= 0:
        raise ValueError("topk must be > 0")

    x = scores_3d.unsqueeze(0).unsqueeze(0)
    suppressed = nms_3d(x, nms_radius=nms_radius).squeeze(0).squeeze(0)

    flat = suppressed.reshape(-1)
    vals, idxs = torch.topk(flat, k=min(topk, flat.numel()))

    d, h, w = [int(s) for s in suppressed.shape]
    peaks = []
    for score, idx in zip(vals, idxs):
        score_v = float(score.item())
        if score_v <= float(score_threshold):
            continue

        i = int(idx.item())
        z = i // (h * w)
        rem = i % (h * w)
        y = rem // w
        x_coord = rem % w

        peaks.append(
            {
                "z": z,
                "y": y,
                "x": x_coord,
                "score": score_v,
            }
        )
    return peaks


def run_section_7_2_assertions():
    torch.manual_seed(123)
    atol = 1e-6

    # 5-2 validation A: 2D NMS keeps only local maxima.
    s2 = torch.zeros(1, 1, 7, 7)
    s2[0, 0, 2, 2] = 0.9
    s2[0, 0, 2, 3] = 0.8
    s2[0, 0, 5, 5] = 0.7
    out2 = nms_2d(s2, nms_radius=1)
    assert abs(out2[0, 0, 2, 2].item() - 0.9) < atol
    assert abs(out2[0, 0, 2, 3].item() - 0.0) < atol
    assert abs(out2[0, 0, 5, 5].item() - 0.7) < atol

    # 5-2 validation B: 3D NMS suppresses nearby weaker peaks.
    s3 = torch.zeros(1, 1, 8, 8, 8)
    s3[0, 0, 3, 4, 5] = 1.0
    s3[0, 0, 3, 4, 6] = 0.95
    s3[0, 0, 1, 1, 1] = 0.6
    out3 = nms_3d(s3, nms_radius=1)
    assert abs(out3[0, 0, 3, 4, 5].item() - 1.0) < atol
    assert abs(out3[0, 0, 3, 4, 6].item() - 0.0) < atol
    assert abs(out3[0, 0, 1, 1, 1].item() - 0.6) < atol

    # 5-2 validation C: top-k extraction returns sorted peak coordinates.
    volume = out3[0, 0]
    peaks = extract_topk_after_nms_3d(volume, nms_radius=1, topk=2, score_threshold=0.1)
    assert len(peaks) == 2
    assert peaks[0]["score"] >= peaks[1]["score"]
    assert peaks[0]["z"] == 3 and peaks[0]["y"] == 4 and peaks[0]["x"] == 5

    print("[5-2/nms2d_nonzero]", int((out2 > 0).sum().item()))
    print("[5-2/nms3d_nonzero]", int((out3 > 0).sum().item()))
    print("[5-2/top1_peak]", peaks[0])


def build_postprocess_cfg(nms_radius=2, topk=1, score_threshold=0.0):
    return SimpleNamespace(
        nms_radius=int(nms_radius),
        topk=int(topk),
        score_threshold=float(score_threshold),
    )
