import random

import torch


def rotate(x, mask=None, dims=((-3, -2), (-3, -1), (-2, -1)), p=1.0):
    """Rotate volumes by random k*90 deg on selected dimension pairs."""
    for d in dims:
        if random.random() < p:
            k = random.randint(0, 3)
            x = torch.rot90(x, k=k, dims=d)
            if mask is not None:
                mask = torch.rot90(mask, k=k, dims=d)

    if mask is not None:
        return x, mask
    return x


def flip_3d(x, mask=None, dims=(-3, -2, -1), p=0.5):
    """Randomly flip along spatial axes."""
    axes = [i for i in dims if random.random() < p]
    if axes:
        x = torch.flip(x, dims=axes)
        if mask is not None:
            mask = torch.flip(mask, dims=axes)

    if mask is not None:
        return x, mask
    return x


def swap_dims(x, mask=None, p=0.5, dims=(-2, -1)):
    """Randomly swap two spatial dimensions."""
    if random.random() < p:
        swap_order = list(dims)
        random.shuffle(swap_order)
        x = x.transpose(*swap_order)
        if mask is not None:
            mask = mask.transpose(*swap_order)

    if mask is not None:
        return x, mask
    return x


def coarse_dropout_3d(x, mask=None, p=0.5, fill_val=0.0, num_holes=(1, 3), hole_range=(8, 64, 64)):
    """Apply random cuboid erasing on 3D volumes."""
    if torch.rand(1).item() >= p:
        if mask is not None:
            return x, mask
        return x

    zs, ys, xs = x.shape[-3:]
    max_d = min(int(hole_range[0]), max(2, zs))
    max_h = min(int(hole_range[1]), max(2, ys))
    max_w = min(int(hole_range[2]), max(2, xs))

    n_low = max(1, int(num_holes[0]))
    n_high = max(n_low + 1, int(num_holes[1]))
    n_holes = torch.randint(low=n_low, high=n_high, size=(1,), device="cpu").item()

    for _ in range(n_holes):
        d = torch.randint(low=2, high=max_d + 1, size=(1,), device="cpu").item()
        h = torch.randint(low=2, high=max_h + 1, size=(1,), device="cpu").item()
        w = torch.randint(low=2, high=max_w + 1, size=(1,), device="cpu").item()

        z0 = torch.randint(low=0, high=zs - d + 1, size=(1,), device="cpu").item()
        y0 = torch.randint(low=0, high=ys - h + 1, size=(1,), device="cpu").item()
        x0 = torch.randint(low=0, high=xs - w + 1, size=(1,), device="cpu").item()

        x[..., z0 : z0 + d, y0 : y0 + h, x0 : x0 + w] = fill_val

    if mask is not None:
        return x, mask
    return x


def run_section_6_3_assertions():
    random.seed(123)
    torch.manual_seed(123)

    x = torch.arange(1 * 1 * 6 * 8 * 10, dtype=torch.float32).reshape(1, 1, 6, 8, 10)
    mask = torch.ones(1, 1, 6, 8, 10)

    # 4-3 validation A: rotate/flip/swap keep tensor/mask shape consistency.
    xr, mr = rotate(x.clone(), mask.clone(), p=1.0)
    xf, mf = flip_3d(xr, mr, p=1.0)
    xs, ms = swap_dims(xf, mf, p=1.0, dims=(-2, -1))
    assert xs.shape[0] == x.shape[0] and xs.shape[1] == x.shape[1]
    assert xs.numel() == x.numel()
    assert ms.shape == xs.shape

    # 4-3 validation B: coarse dropout modifies content while keeping shape.
    xd, md = coarse_dropout_3d(
        x.clone(),
        mask.clone(),
        p=1.0,
        fill_val=0.0,
        num_holes=(1, 3),
        hole_range=(4, 4, 4),
    )
    assert xd.shape == x.shape
    assert md.shape == mask.shape
    assert (xd == 0).sum().item() > (x == 0).sum().item()

    # 4-3 validation C: no-mask path returns tensor only.
    x_nomask = coarse_dropout_3d(x.clone(), mask=None, p=1.0, num_holes=(1, 2), hole_range=(3, 3, 3))
    assert isinstance(x_nomask, torch.Tensor)
    assert x_nomask.shape == x.shape

    print("[4-3/rotate_flip_swap_shape]", tuple(xs.shape))
    print("[4-3/coarse_dropout_zero_count]", int((xd == 0).sum().item()))
    print("[4-3/no_mask_shape]", tuple(x_nomask.shape))
