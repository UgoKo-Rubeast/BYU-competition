import random

import torch
import torch.nn as nn
from torch.distributions import Beta


class Mixup(nn.Module):
    def __init__(self, beta, mixadd=False):
        super().__init__()
        self.beta_distribution = Beta(beta, beta)
        self.mixadd = bool(mixadd)

    def forward(self, X, Y, Z=None):
        b = X.shape[0]
        coeffs = self.beta_distribution.rsample(torch.Size((b,))).to(X.device)

        X_coeffs = coeffs.view((-1,) + (1,) * (X.ndim - 1))
        Y_coeffs = coeffs.view((-1,) + (1,) * (Y.ndim - 1))

        perm = torch.randperm(X.size(0), device=X.device)
        X_perm = X[perm]
        Y_perm = Y[perm]

        X = X_coeffs * X + (1.0 - X_coeffs) * X_perm

        if self.mixadd:
            Y = (Y + Y_perm).clip(0, 1)
        else:
            Y = Y_coeffs * Y + (1.0 - Y_coeffs) * Y_perm

        if Z is not None:
            return X, Y, Z
        return X, Y


class CutmixSimple(nn.Module):
    def __init__(self, beta=5.0, dims=(-2, -1)):
        super().__init__()
        if not all(_ < 0 for _ in dims):
            raise ValueError("dims must be negatively indexed.")
        self.beta_distribution = Beta(beta, beta)
        self.dims = tuple(dims)

    def forward(self, X, Y, Z=None):
        cut_ratio = self.beta_distribution.sample().item()

        perm = torch.randperm(X.size(0), device=X.device)
        X_perm = X[perm]
        Y_perm = Y[perm]

        axis = random.choice(self.dims)

        cutoff_X = int(cut_ratio * X.shape[axis])
        cutoff_Y = int(cut_ratio * Y.shape[axis])

        if axis == -1:
            X[..., :cutoff_X] = X_perm[..., :cutoff_X]
            Y[..., :cutoff_Y] = Y_perm[..., :cutoff_Y]
        elif axis == -2:
            X[..., :cutoff_X, :] = X_perm[..., :cutoff_X, :]
            Y[..., :cutoff_Y, :] = Y_perm[..., :cutoff_Y, :]
        else:
            raise ValueError("CutmixSimple: Axis not implemented.")

        if Z is not None:
            return X, Y, Z
        return X, Y


def run_section_6_4_assertions():
    torch.manual_seed(123)
    random.seed(123)

    X = torch.randn(4, 1, 8, 16, 16)
    Y = (torch.rand(4, 1, 8, 16, 16) > 0.5).float()

    # 4-4 validation A: mixup keeps shape and produces mixed values.
    mixup = Mixup(beta=0.4)
    Xm, Ym = mixup(X.clone(), Y.clone())
    assert Xm.shape == X.shape
    assert Ym.shape == Y.shape
    assert not torch.allclose(Xm, X)

    # 4-4 validation B: cutmix keeps shape and replaces partial regions.
    cutmix = CutmixSimple(beta=3.0, dims=(-2, -1))
    Xc, Yc = cutmix(X.clone(), Y.clone())
    assert Xc.shape == X.shape
    assert Yc.shape == Y.shape
    assert not torch.allclose(Xc, X)

    # 4-4 validation C: mixadd path stays in [0,1] range.
    mixadd = Mixup(beta=0.7, mixadd=True)
    _, Ym_add = mixadd(X.clone(), Y.clone())
    assert float(Ym_add.min()) >= 0.0
    assert float(Ym_add.max()) <= 1.0

    print("[4-4/mixup_shape]", tuple(Xm.shape), tuple(Ym.shape))
    print("[4-4/cutmix_shape]", tuple(Xc.shape), tuple(Yc.shape))
    print("[4-4/mixadd_range]", float(Ym_add.min()), float(Ym_add.max()))
