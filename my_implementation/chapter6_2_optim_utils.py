from types import SimpleNamespace

import torch
import torch.nn as nn


def calc_grad_norm(parameters, norm_type=2.0):
    """Compute total gradient norm; returns None if nan/inf detected."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)

    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type,
    )
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        return None
    return total_norm


def get_optimizer(model, cfg):
    """WBS 4-2: optimizer builder."""
    lr = float(getattr(cfg, "lr", 1e-4))
    weight_decay = float(getattr(cfg, "weight_decay", 1e-4))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, cfg, n_steps):
    """WBS 4-2: scheduler builder compatible with past implementation options."""
    scheduler_name = getattr(cfg, "scheduler", "Constant")
    if scheduler_name == "Constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    if scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(n_steps),
            eta_min=float(getattr(cfg, "lr_min", 1e-5)),
        )
    raise ValueError(f"{scheduler_name} is not a valid scheduler.")


def clip_grad_and_measure(model, max_norm, norm_type=2.0):
    """Measure grad norm before/after clipping and return tuple of tensors/None."""
    before = calc_grad_norm(model.parameters(), norm_type=norm_type)
    if max_norm is not None and float(max_norm) > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm), norm_type=float(norm_type))
    after = calc_grad_norm(model.parameters(), norm_type=norm_type)
    return before, after


def run_section_6_2_assertions():
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(1, 2, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    # 4-2 validation A: optimizer + constant scheduler build.
    model = TinyNet()
    cfg_const = SimpleNamespace(lr=1e-3, weight_decay=1e-2, scheduler="Constant", lr_min=1e-5)
    opt = get_optimizer(model, cfg_const)
    sch_const = get_scheduler(opt, cfg_const, n_steps=10)
    assert isinstance(opt, torch.optim.AdamW)
    assert isinstance(sch_const, torch.optim.lr_scheduler.LambdaLR)

    # 4-2 validation B: cosine scheduler builds and updates LR.
    cfg_cos = SimpleNamespace(lr=1e-3, weight_decay=0.0, scheduler="CosineAnnealingLR", lr_min=1e-5)
    opt_cos = get_optimizer(model, cfg_cos)
    sch_cos = get_scheduler(opt_cos, cfg_cos, n_steps=8)
    lr_before = opt_cos.param_groups[0]["lr"]
    sch_cos.step()
    lr_after = opt_cos.param_groups[0]["lr"]
    assert isinstance(sch_cos, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert lr_after <= lr_before

    # 4-2 validation C: grad norm tracking and clipping.
    x = torch.randn(2, 1, 8, 8, 8)
    target = torch.randn(2, 2, 8, 8, 8)
    opt.zero_grad(set_to_none=True)
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()

    before, after = clip_grad_and_measure(model, max_norm=0.1, norm_type=2.0)
    assert before is not None and after is not None
    assert float(after) <= float(before) + 1e-8

    print("[4-2/optimizer]", type(opt).__name__)
    print("[4-2/scheduler_constant]", type(sch_const).__name__)
    print("[4-2/scheduler_cosine]", type(sch_cos).__name__)
    print("[4-2/lr_before_after]", float(lr_before), float(lr_after))
    print("[4-2/grad_norm_before_after]", float(before), float(after))
