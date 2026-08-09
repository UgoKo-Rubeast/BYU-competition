import json
from abc import ABC, abstractmethod
from types import SimpleNamespace


def _is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except TypeError:
        return False


def flatten_config(cfg_dict, parent_key=""):
    flat = {}
    for key, value in cfg_dict.items():
        now_key = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, dict):
            flat.update(flatten_config(value, parent_key=now_key))
            continue

        if _is_json_serializable(value):
            flat[now_key] = value
        else:
            flat[now_key] = str(value)
    return flat


class BaseLogger(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.hparams = flatten_config(vars(cfg)) if hasattr(cfg, "__dict__") else {}

    @abstractmethod
    def log(self, metrics=None, commit=True):
        raise NotImplementedError

    @abstractmethod
    def finish(self):
        raise NotImplementedError


class NoLogger(BaseLogger):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.history = []

    def log(self, metrics=None, commit=True):
        if metrics is None:
            metrics = {}
        self.history.append({"metrics": dict(metrics), "commit": bool(commit)})

    def finish(self):
        return


class WandbLogger(BaseLogger):
    def __init__(self, cfg, wandb_module):
        super().__init__(cfg)
        self._wandb = wandb_module

        init_kwargs = {
            "project": str(getattr(cfg, "project", "byu-competition")),
            "config": self.hparams,
        }

        run_name = getattr(cfg, "run_name", None)
        if run_name:
            init_kwargs["name"] = str(run_name)

        group = getattr(cfg, "group", None)
        if group:
            init_kwargs["group"] = str(group)

        self._wandb.init(**init_kwargs)

    def log(self, metrics=None, commit=True):
        if metrics is None:
            metrics = {}
        self._wandb.log(dict(metrics), commit=bool(commit))

    def finish(self):
        self._wandb.finish()


def _resolve_wandb_module(wandb_module=None):
    if wandb_module is not None:
        return wandb_module

    try:
        import wandb  # type: ignore

        return wandb
    except Exception:
        return None


def get_logger(cfg, wandb_module=None):
    """WBS 6-3: create a process-safe logger with WandB fallback."""
    logger_name = str(getattr(cfg, "logger", "none")).lower()
    local_rank = int(getattr(cfg, "local_rank", 0))
    is_main_process = local_rank in {0, -1}

    if (not is_main_process) or logger_name != "wandb":
        return NoLogger(cfg)

    resolved = _resolve_wandb_module(wandb_module=wandb_module)
    if resolved is None:
        return NoLogger(cfg)

    return WandbLogger(cfg, wandb_module=resolved)


class _FakeWandb:
    def __init__(self):
        self.init_calls = []
        self.log_calls = []
        self.finish_calls = 0

    def init(self, **kwargs):
        self.init_calls.append(kwargs)

    def log(self, metrics, commit=True):
        self.log_calls.append({"metrics": dict(metrics), "commit": bool(commit)})

    def finish(self):
        self.finish_calls += 1


def run_section_8_3_assertions():
    cfg = SimpleNamespace(
        project="byu-demo",
        logger="wandb",
        local_rank=0,
        run_name="exp-6-3",
        group="fold-0",
        nested={"batch_size": 2, "depth": 16},
    )
    fake_wandb = _FakeWandb()

    # 6-3 validation A: main process + wandb logger uses WandB client.
    logger = get_logger(cfg, wandb_module=fake_wandb)
    assert isinstance(logger, WandbLogger)
    assert len(fake_wandb.init_calls) == 1
    init_cfg = fake_wandb.init_calls[0]["config"]
    assert init_cfg["nested.batch_size"] == 2
    assert init_cfg["nested.depth"] == 16

    # 6-3 validation B: metrics and finish are delegated.
    logger.log({"train/loss": 0.25}, commit=True)
    logger.log({"val/fbeta": 0.8}, commit=False)
    logger.finish()
    assert len(fake_wandb.log_calls) == 2
    assert fake_wandb.log_calls[0]["metrics"]["train/loss"] == 0.25
    assert fake_wandb.log_calls[1]["commit"] is False
    assert fake_wandb.finish_calls == 1

    # 6-3 validation C: non-main process is NoLogger.
    cfg_non_main = SimpleNamespace(logger="wandb", local_rank=1)
    logger_non_main = get_logger(cfg_non_main, wandb_module=fake_wandb)
    assert isinstance(logger_non_main, NoLogger)

    # 6-3 validation D: non-wandb mode is NoLogger.
    cfg_none = SimpleNamespace(logger="none", local_rank=0)
    logger_none = get_logger(cfg_none, wandb_module=fake_wandb)
    assert isinstance(logger_none, NoLogger)

    print("[6-3/wandb_init_calls]", len(fake_wandb.init_calls))
    print("[6-3/wandb_log_calls]", len(fake_wandb.log_calls))
    print("[6-3/non_main_type]", type(logger_non_main).__name__)
    print("[6-3/none_type]", type(logger_none).__name__)
