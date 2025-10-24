import os
import json
import random

import torch
import numpy as np


def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_json(file_path: str):
    try:
        with open(file_path, "rb") as reader:
            data = json.load(reader)
        return data
    except IOError as e:
        raise IOError(f"Failed to read file {file_path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}") from e


def write_json(data, file_path: str):
    with open(file_path, "w", encoding="utf-8") as writer:
        json.dump(data, writer, indent=2)
