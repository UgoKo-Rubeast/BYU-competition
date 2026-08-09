from types import SimpleNamespace


def _normalize_gpus(gpus):
    if isinstance(gpus, (list, tuple)):
        ids = [str(x).strip() for x in gpus if str(x).strip() != ""]
    else:
        ids = [x.strip() for x in str(gpus).split(",") if x.strip() != ""]

    if len(ids) == 0:
        raise ValueError("gpus must contain at least one gpu id")

    for gid in ids:
        if not gid.isdigit():
            raise ValueError(f"invalid gpu id: {gid}")
    return ids


def _format_overrides(overrides):
    if not overrides:
        return ""

    parts = []
    for key, value in overrides.items():
        parts.append(f"{key}={value}")
    return " " + " ".join(parts)


def build_torchrun_command(cfg):
    """WBS 6-2: build a robust multi-GPU torchrun command string."""
    gpus = _normalize_gpus(getattr(cfg, "gpus", "0"))
    nproc_per_node = int(getattr(cfg, "nproc_per_node", len(gpus)))
    nnodes = int(getattr(cfg, "nnodes", 1))
    node_rank = int(getattr(cfg, "node_rank", 0))
    master_addr = str(getattr(cfg, "master_addr", "127.0.0.1"))
    master_port = int(getattr(cfg, "master_port", 29500))

    train_script = str(getattr(cfg, "train_script", "train.py"))
    config_name = str(getattr(cfg, "config", "r3d200"))
    extra_overrides = dict(getattr(cfg, "overrides", {}))

    env_part = f"CUDA_VISIBLE_DEVICES={','.join(gpus)}"
    run_part = (
        "torchrun "
        f"--nproc_per_node={nproc_per_node} "
        f"--nnodes={nnodes} "
        f"--node_rank={node_rank} "
        f"--master_addr={master_addr} "
        f"--master_port={master_port}"
    )

    train_part = f"python {train_script} -C={config_name}"
    train_part += _format_overrides(extra_overrides)

    return f"{env_part} {run_part} {train_part}".strip()


def build_run_sh_text(cfg):
    command = build_torchrun_command(cfg)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# WBS 6-2: multi-GPU training launcher",
        f"{command}",
        "",
    ]
    return "\n".join(lines)


def validate_run_sh_text(script_text):
    checks = {
        "has_shebang": script_text.startswith("#!/usr/bin/env bash"),
        "has_strict_mode": "set -euo pipefail" in script_text,
        "has_torchrun": "torchrun" in script_text,
        "has_cuda_visible_devices": "CUDA_VISIBLE_DEVICES=" in script_text,
        "has_train_py": "python" in script_text and "train.py" in script_text,
    }
    checks["is_valid"] = all(checks.values())
    return checks


def run_section_8_2_assertions():
    cfg = SimpleNamespace(
        gpus="0,1",
        nproc_per_node=2,
        nnodes=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29510,
        train_script="train.py",
        config="r3d200",
        overrides={"epochs": 2, "fold": 999, "save_weights": True},
    )

    cmd = build_torchrun_command(cfg)
    assert "CUDA_VISIBLE_DEVICES=0,1" in cmd
    assert "--nproc_per_node=2" in cmd
    assert "--master_port=29510" in cmd
    assert "python train.py -C=r3d200" in cmd
    assert "epochs=2" in cmd and "fold=999" in cmd

    sh_text = build_run_sh_text(cfg)
    check = validate_run_sh_text(sh_text)
    assert check["is_valid"]

    print("[6-2/cmd]", cmd)
    print("[6-2/valid]", check["is_valid"])
