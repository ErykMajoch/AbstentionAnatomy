import yaml
import torch
import numpy as np
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as config:
        return yaml.safe_load(config)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: dict) -> torch.device:
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device)


def get_dtype(config: dict) -> torch.dtype:
    dtype_str = config.get("dtype", "float16")
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_str]


def setup(config_path: str | Path | None) -> dict:
    config = load_config(config_path)
    set_seed(config["seed"])
    config["_device"] = get_device(config)
    config["_dtype"] = get_dtype(config)
    return config
