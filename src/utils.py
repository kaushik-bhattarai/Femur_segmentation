"""Small shared helpers: config loading, seeding, device, JSON I/O."""
import json
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path, overrides=None):
    """Load a YAML config and apply overrides given as 'section.key=value' strings."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, value = item.split("=", 1)
        value = yaml.safe_load(value)  # '5' -> 5, 'true' -> True, '[1,2]' -> list
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[parts[-1]] = value
    return cfg


def add_common_args(parser):
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--opt", action="append", default=[],
                        help="override a config value, e.g. --opt train.epochs=5")
    return parser


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_meta(case_dir):
    return read_json(Path(case_dir) / "meta.json")
