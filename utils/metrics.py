from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    running_loss = 0.0
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            running_loss += loss.item() * x_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            y_pred.append(preds)
            y_true.append(y_batch.long().cpu().numpy())

    return build_epoch_metrics(y_true, y_pred, running_loss, len(loader.dataset))


def predict_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            logits = model(x_batch)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            y_pred.append(preds)
            y_true.append(y_batch.long().cpu().numpy())

    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "precision": float(precision_score(y_true_np, y_pred_np, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred_np, zero_division=0)),
        "f1": float(f1_score(y_true_np, y_pred_np, zero_division=0)),
        "positive_ratio": float(y_true_np.mean()),
        "test_size": int(len(y_true_np)),
    }


def build_epoch_metrics(
    y_true: List[np.ndarray],
    y_pred: List[np.ndarray],
    running_loss: float,
    dataset_size: int,
) -> EpochMetrics:
    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    avg_loss = running_loss / dataset_size
    return EpochMetrics(
        loss=avg_loss,
        accuracy=accuracy_score(y_true_np, y_pred_np),
        precision=precision_score(y_true_np, y_pred_np, zero_division=0),
        recall=recall_score(y_true_np, y_pred_np, zero_division=0),
        f1=f1_score(y_true_np, y_pred_np, zero_division=0),
    )


def metrics_to_dict(metrics: EpochMetrics) -> Dict[str, float]:
    return asdict(metrics)
