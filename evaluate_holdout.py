import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.lstm_classifier import build_lstm_classifier
from utils.checkpoint import load_checkpoint
from utils.config import parse_args_with_config
from utils.dataset import (
    NpySequenceDataset,
    compute_normalization_stats,
    sample_holdout_indices,
    sample_indices,
    split_train_val,
)
from utils.metrics import predict_classifier
from utils.runtime import get_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved LSTM checkpoint on a stratified holdout set")
    parser.add_argument("--config", default=None, help="Path to a JSON config file.")
    parser.add_argument("--checkpoint", default="artifacts/best_lstm_confirm_800k.pt")
    parser.add_argument("--output", default="artifacts/test_metrics_800k_holdout.json")
    parser.add_argument("--test-size", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser


def parse_args() -> argparse.Namespace:
    return parse_args_with_config(build_parser)


def evaluate_holdout(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    train_args = checkpoint["args"]
    seed = int(train_args["seed"])
    set_seed(seed)

    labels = np.load(train_args["y_path"], mmap_mode="r")
    sampled_indices = sample_indices(labels, int(train_args["sample_size"]), seed)
    sampled_labels = np.asarray(labels[sampled_indices], dtype=np.int64)
    train_idx, _ = split_train_val(sampled_indices, sampled_labels, float(train_args["val_ratio"]), seed)
    holdout_idx = sample_holdout_indices(labels, sampled_indices, args.test_size, seed + 1)

    mean = std = None
    if train_args.get("standardize", False):
        mean, std = compute_normalization_stats(
            train_args["x_path"],
            train_idx,
            int(train_args["max_stats_samples"]),
            random_seed=seed,
        )

    x_mem = np.load(train_args["x_path"], mmap_mode="r")
    model = build_lstm_classifier(
        input_size=int(x_mem.shape[-1]),
        hidden_size=int(train_args["hidden_size"]),
        num_layers=int(train_args["num_layers"]),
        dropout=float(train_args["dropout"]),
        bidirectional=bool(train_args["bidirectional"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = get_device()
    model.to(device)

    loader = DataLoader(
        NpySequenceDataset(train_args["x_path"], train_args["y_path"], holdout_idx, mean, std),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    metrics = predict_classifier(model, loader, device)
    metrics["checkpoint"] = args.checkpoint
    metrics["test_sampling"] = "stratified holdout from samples not used in the training run"
    metrics["seed"] = seed + 1

    with open(args.output, "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False))


def main() -> None:
    evaluate_holdout(parse_args())


if __name__ == "__main__":
    main()
