import os
from typing import Any, Dict

import torch
from torch import nn

from utils.metrics import EpochMetrics, metrics_to_dict


def save_checkpoint(
    model: nn.Module,
    args: Dict[str, Any],
    train_size: int,
    val_size: int,
    best_metrics: EpochMetrics,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "args": args,
        "train_size": int(train_size),
        "val_size": int(val_size),
        "best_metrics": metrics_to_dict(best_metrics),
    }
    torch.save(payload, output_path)


def load_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(checkpoint_path, map_location=map_location)
