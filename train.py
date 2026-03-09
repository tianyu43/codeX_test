import argparse
import json
import math
import os
from typing import Dict, List, Tuple, Union

import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch

from models.lstm_classifier import build_lstm_classifier
from utils.checkpoint import save_checkpoint
from utils.config import parse_args_with_config
from utils.dataset import (
    NpySequenceDataset,
    compute_normalization_stats,
    sample_indices,
    split_train_val,
)
from utils.metrics import EpochMetrics, build_epoch_metrics, evaluate_classifier
from utils.runtime import get_device, set_seed


HistoryRow = Dict[str, Union[float, int]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LSTM classifier for remote sensing time-series data")
    parser.add_argument("--config", default=None, help="Path to a JSON config file.")
    parser.add_argument("--x-path", default="data/x_sr_s.npy")
    parser.add_argument("--y-path", default="data/label.npy")
    parser.add_argument("--sample-size", type=int, default=200000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-stats-samples", type=int, default=50000)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/lstm_classifier.pt")
    parser.add_argument("--history-json", default="artifacts/lstm_history.json")
    return parser


def parse_args() -> argparse.Namespace:
    return parse_args_with_config(build_parser)


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    labels = np.load(args.y_path, mmap_mode="r")
    sampled_indices = sample_indices(labels, args.sample_size, args.seed)
    sampled_labels = np.asarray(labels[sampled_indices], dtype=np.int64)
    train_idx, val_idx = split_train_val(sampled_indices, sampled_labels, args.val_ratio, args.seed)

    mean = std = None
    if args.standardize:
        mean, std = compute_normalization_stats(
            args.x_path,
            train_idx,
            args.max_stats_samples,
            random_seed=args.seed,
        )

    train_ds = NpySequenceDataset(args.x_path, args.y_path, train_idx, mean, std)
    val_ds = NpySequenceDataset(args.x_path, args.y_path, val_idx, mean, std)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_idx, val_idx


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> EpochMetrics:
    model.train()
    running_loss = 0.0
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)
        preds = (torch.sigmoid(logits.detach()) >= 0.5).long().cpu().numpy()
        y_pred.append(preds)
        y_true.append(y_batch.long().cpu().numpy())

    return build_epoch_metrics(y_true, y_pred, running_loss, len(loader.dataset))


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    x_mem = np.load(args.x_path, mmap_mode="r")
    input_size = int(x_mem.shape[-1])

    train_loader, val_loader, train_idx, val_idx = build_loaders(args)
    train_labels = np.load(args.y_path, mmap_mode="r")[train_idx]
    pos_count = float(train_labels.sum())
    neg_count = float(len(train_labels) - pos_count)
    pos_weight = torch.tensor(max(neg_count / max(pos_count, 1.0), 1.0), device=device)

    model = build_lstm_classifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[HistoryRow] = []
    best_f1 = -math.inf
    best_metrics = EpochMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    print(f"device={device}")
    print(f"train_size={len(train_idx)} val_size={len(val_idx)}")
    print(f"positive_ratio_train={float(train_labels.mean()):.6f}")
    print(f"model_params={sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        val_metrics = evaluate_classifier(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
            "train_f1": train_metrics.f1,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.accuracy,
            "val_precision": val_metrics.precision,
            "val_recall": val_metrics.recall,
            "val_f1": val_metrics.f1,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=True))

        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
            best_metrics = val_metrics
            save_checkpoint(
                model,
                vars(args),
                len(train_idx),
                len(val_idx),
                best_metrics,
                args.output,
            )

    os.makedirs(os.path.dirname(args.history_json) or ".", exist_ok=True)
    with open(args.history_json, "w", encoding="utf-8") as file_obj:
        json.dump(
            {"best_metrics": vars(best_metrics), "history": history},
            file_obj,
            ensure_ascii=False,
            indent=2,
        )

    print(f"best_val_f1={best_metrics.f1:.6f}")
    print(f"checkpoint={args.output}")
    print(f"history={args.history_json}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
