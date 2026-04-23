from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    ndcg_k,
    hit_k
)


class Args:
    """Helper to pass config parameters to SASRec model."""
    def __init__(self, config: Dict[str, Any]):
        self.hidden_units = config["model"]["hidden_units"]
        self.num_blocks = config["model"]["num_blocks"]
        self.num_heads = config["model"]["num_heads"]
        self.dropout_rate = config["model"]["dropout_rate"]
        self.maxlen = config["model"]["maxlen"]
        self.device = torch.device(config["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.norm_first = config["model"].get("norm_first", False)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    checkpoint_dir: str,
    filename: str,
    metrics: Optional[Dict[str, float]] = None
):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_dir+"__weights", exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    path_weights_only = os.path.join(checkpoint_dir+"__weights", filename)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if metrics is not None:
        state.update(metrics)

    torch.save(state, path)


def plot_training_curves(
    history: Dict[str, Any],
    save_path: str,
    title: str = "Training Curves",
) -> None:
    """Plot loss and metrics curves for train/val/test and save to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot generation")
        return

    train_epochs = [e for e, _ in history.get("train_losses", [])]
    train_losses = [l for _, l in history.get("train_losses", [])]
    val_epochs = history.get("val_epochs", [])
    val_ndcg = history.get("val_ndcg", [])
    val_hit = history.get("val_hit", [])
    test_ndcg = history.get("test_ndcg")
    test_hit = history.get("test_hit")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # --- Loss ---
    if train_losses:
        axes[0].plot(train_epochs, train_losses, "b-o", markersize=3, linewidth=1.5, label="Train")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].set_visible(False)

    # --- NDCG@10 ---
    if val_ndcg:
        axes[1].plot(val_epochs, val_ndcg, "g-o", markersize=4, linewidth=1.5, label="Val")
        if test_ndcg is not None:
            axes[1].axhline(
                y=test_ndcg, color="r", linestyle="--", linewidth=1.5,
                label=f"Test = {test_ndcg:.4f}",
            )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("NDCG@10")
        axes[1].set_title("NDCG@10 (Val / Test)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    # --- Hit@10 ---
    if val_hit:
        axes[2].plot(val_epochs, val_hit, "m-o", markersize=4, linewidth=1.5, label="Val")
        if test_hit is not None:
            axes[2].axhline(
                y=test_hit, color="r", linestyle="--", linewidth=1.5,
                label=f"Test = {test_hit:.4f}",
            )
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Hit@10")
        axes[2].set_title("Hit@10 (Val / Test)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


def plot_adaround_metrics(
    ndcg_val: float,
    hit_val: float,
    ndcg_test: float,
    hit_test: float,
    save_path: str,
    title: str = "AdaRound Metrics",
) -> None:
    """Plot AdaRound val/test metrics as a grouped bar chart and save to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot generation")
        return

    metrics = ["NDCG@10", "Hit@10"]
    val_values = [ndcg_val, hit_val]
    test_values = [ndcg_test, hit_test]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, val_values, width, label="Val", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, test_values, width, label="Test", color="salmon", alpha=0.85)

    offset = max(val_values + test_values) * 0.015
    for bar in (*bars1, *bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + offset,
            f"{h:.4f}", ha="center", va="bottom", fontsize=9,
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"AdaRound metrics chart saved to {save_path}")


def evaluate(
    model: nn.Module,
    dataset: List,
    args: Args,
    logger=None,
    epoch=0
) -> Tuple[float, float]:
    """Evaluate model (NDCG@10 and Hit@10)."""
    model.eval()
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    users = range(1, usernum + 1)

    for u in tqdm(users, desc="Evaluating", leave=False):
        if len(user_train[u]) < 1 or len(user_valid[u]) < 1:
            continue
        seq = user_train[u]
        target_item = user_valid[u][0]

        seq_len = len(seq)
        if seq_len >= args.maxlen:
            seq = seq[-args.maxlen:]
        else:
            seq = [0] * (args.maxlen - seq_len) + seq

        log_seqs = torch.tensor([seq], dtype=torch.long, device=args.device)
        item_indices = list(range(1, itemnum + 1))

        with torch.no_grad():
            predictions = model.predict(
                user_ids=None,
                log_seqs=log_seqs,
                item_indices=item_indices
            )
            predictions = -predictions[0] # negative for ascending sort

            _, indices = torch.topk(predictions, 10, largest=False)
            rank_list = [item_indices[i] for i in indices.cpu().numpy()]

            NDCG += ndcg_k([target_item], rank_list, k=10)
            HT += hit_k([target_item], rank_list, k=10)
            valid_user += 1

    if valid_user == 0:
        return 0.0, 0.0

    avg_ndcg = NDCG / valid_user
    avg_hit = HT / valid_user

    if logger:
        logger.report_scalar("NDCG@10", "Val", value=avg_ndcg, iteration=epoch)
        logger.report_scalar("Hit@10", "Val", value=avg_hit, iteration=epoch)

    return avg_ndcg, avg_hit


def evaluate_test(
    model: nn.Module,
    dataset: List,
    args: Args,
    logger=None,
    epoch: int = 0,
) -> Tuple[float, float]:
    """Final testing on user_test split (NDCG@10 and Hit@10)."""
    model.eval()
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    users = range(1, usernum + 1)

    for u in tqdm(users, desc="Testing", leave=False):
        if len(user_train[u]) < 1 or len(user_test[u]) < 1:
            continue

        seq = user_train[u]
        target_item = user_test[u][0]

        seq_len = len(seq)
        if seq_len >= args.maxlen:
            seq = seq[-args.maxlen:]
        else:
            seq = [0] * (args.maxlen - seq_len) + seq

        log_seqs = torch.tensor([seq], dtype=torch.long, device=args.device)
        item_indices = list(range(1, itemnum + 1))

        with torch.no_grad():
            predictions = model.predict(
                user_ids=None,
                log_seqs=log_seqs,
                item_indices=item_indices,
            )
            predictions = -predictions[0]  # negative for ascending sort

            _, indices = torch.topk(predictions, 10, largest=False)
            rank_list = [item_indices[i] for i in indices.cpu().numpy()]

            NDCG += ndcg_k([target_item], rank_list, k=10)
            HT += hit_k([target_item], rank_list, k=10)
            valid_user += 1

    if valid_user == 0:
        return 0.0, 0.0

    avg_ndcg = NDCG / valid_user
    avg_hit = HT / valid_user

    if logger:
        logger.report_scalar("NDCG@10", "Test", value=avg_ndcg, iteration=epoch)
        logger.report_scalar("Hit@10", "Test", value=avg_hit, iteration=epoch)

    return avg_ndcg, avg_hit


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    logger=None,
    epoch=0
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        u, seq, pos, neg = [x.to(device) for x in batch]

        pos_logits, neg_logits = model(u, seq, pos, neg)

        pos_labels = torch.ones(pos_logits.shape, device=device)
        neg_labels = torch.zeros(neg_logits.shape, device=device)

        optimizer.zero_grad()
        indices = np.where(pos.cpu().numpy() != 0) # Ignore padding
        loss = criterion(pos_logits[indices], pos_labels[indices])
        loss += criterion(neg_logits[indices], neg_labels[indices])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    if logger:
        logger.report_scalar("Loss", "Train", value=avg_loss, iteration=epoch)

    return avg_loss


def train_fp32(
    model: nn.Module,
    train_loader: DataLoader,
    dataset: List,
    config: Dict[str, Any],
    criterion: nn.Module,
    args: Args,
    checkpoint_dir: str,
    save_name: str = "sasrec_fp32.pth",
    logger=None
):
    """Train FP32 model."""
    print("Starting FP32 training...")

    optimizer = Adam(
        model.parameters(),
        lr=config["optimization"]["lr"],
        betas=tuple(config["optimization"].get("betas", (0.9, 0.98))),
        weight_decay=config["optimization"].get("weight_decay", 0.0)
    )

    epochs = config["training"]["epochs"]
    eval_interval = config["training"].get("eval_interval", 20)
    best_ndcg = 0.0
    run_name = config["experiment"]["run_name"]

    history: Dict[str, Any] = {
        "train_losses": [],
        "val_epochs": [],
        "val_ndcg": [],
        "val_hit": [],
    }

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, args.device, logger=logger, epoch=epoch)
        history["train_losses"].append((epoch, loss))

        if epoch % eval_interval == 0:
            ndcg, ht = evaluate(model, dataset, args, logger=logger, epoch=epoch)
            history["val_epochs"].append(epoch)
            history["val_ndcg"].append(ndcg)
            history["val_hit"].append(ht)
            print(f"[FP32] Epoch {epoch}/{epochs} | Loss: {loss:.4f} | NDCG@10: {ndcg:.4f} | Hit@10: {ht:.4f}")

            results = {"ndcg": ndcg, "hit": ht}
            with open(Path(config["paths"]["results_dir"]) / f"{run_name}_results.json", "w") as f:
                json.dump(results, f, indent=2)

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                save_checkpoint(
                    model, optimizer, epoch, checkpoint_dir, save_name,
                    metrics={"ndcg": ndcg, "hit": ht}
                )
                print(f"Saved best FP32 checkpoint (NDCG: {ndcg:.4f})")
        else:
            print(f"[FP32] Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

    print(f"FP32 training done. Best NDCG (Val): {best_ndcg:.4f}")

    ckpt_path = os.path.join(checkpoint_dir, save_name)
    if os.path.exists(ckpt_path):
        print(f"Loading best FP32 checkpoint from {ckpt_path} for final testing...")
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    print("Running final test evaluation (user_test split)...")
    test_ndcg, test_hit = evaluate_test(model, dataset, args, logger=None, epoch=0)
    print(f"[FP32][Test] NDCG@10: {test_ndcg:.4f} | Hit@10: {test_hit:.4f}")

    history["test_ndcg"] = test_ndcg
    history["test_hit"] = test_hit

    results_path = Path(config["paths"]["results_dir"]) / f"{run_name}_results.json"
    results: Dict[str, Any] = {}
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}
    results.update({
        "test_ndcg": test_ndcg,
        "test_hit": test_hit,
        "history": {
            "train_losses": history["train_losses"],
            "val_epochs": history["val_epochs"],
            "val_ndcg": history["val_ndcg"],
            "val_hit": history["val_hit"],
        },
    })
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    plot_path = str(Path(config["paths"]["results_dir"]) / f"{run_name}_training_curves.png")
    plot_training_curves(history, plot_path, title=f"FP32 Training — {run_name}")


def train_qat(
    model: nn.Module,
    train_loader: DataLoader,
    dataset: List,
    config: Dict[str, Any],
    criterion: nn.Module,
    args: Args,
    strategy_name: str,
    quant_config: Dict[str, Any],
    checkpoint_dir: str,
    save_name: str = "sasrec_qat.pth",
    logger=None
):
    """Train QAT model."""
    print(f"Starting QAT with {strategy_name.upper()}...")

    model.prepare_quant(strategy_name, quant_config)

    print("Running dummy pass to initialize quantization parameters...")
    model.train()
    with torch.no_grad():
        for batch in train_loader:
            u, seq, pos, neg = [x.to(args.device) for x in batch]
            model(u, seq, pos, neg)
            break

    optimizer = Adam(
        model.parameters(),
        lr=config["optimization"]["lr"],
        betas=tuple(config["optimization"].get("betas", (0.9, 0.98))),
        weight_decay=config["optimization"].get("weight_decay", 0.0)
    )

    epochs = config["training"]["epochs"]
    eval_interval = config["training"].get("eval_interval", 20)
    best_ndcg = 0.0
    run_name = config["experiment"]["run_name"]

    history: Dict[str, Any] = {
        "train_losses": [],
        "val_epochs": [],
        "val_ndcg": [],
        "val_hit": [],
    }

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, args.device, logger=logger, epoch=epoch)
        history["train_losses"].append((epoch, loss))

        if epoch % eval_interval == 0:
            ndcg, ht = evaluate(model, dataset, args, logger=logger, epoch=epoch)
            history["val_epochs"].append(epoch)
            history["val_ndcg"].append(ndcg)
            history["val_hit"].append(ht)
            print(f"[QAT {strategy_name}] Epoch {epoch}/{epochs} | Loss: {loss:.4f} | NDCG@10: {ndcg:.4f} | Hit@10: {ht:.4f}")

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                save_checkpoint(
                    model, optimizer, epoch, checkpoint_dir, save_name,
                    metrics={"ndcg": ndcg, "hit": ht}
                )
                print(f"Saved best QAT checkpoint (NDCG: {ndcg:.4f})")
        else:
             print(f"[QAT {strategy_name}] Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

    print(f"QAT ({strategy_name}) done. Best NDCG (Val): {best_ndcg:.4f}")

    ckpt_path = os.path.join(checkpoint_dir, save_name)
    if os.path.exists(ckpt_path):
        print(f"Loading best QAT checkpoint from {ckpt_path} for final testing...")
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    print("Running final test evaluation (user_test split)...")
    test_ndcg, test_hit = evaluate_test(model, dataset, args, logger=None, epoch=0)
    print(f"[QAT {strategy_name}][Test] NDCG@10: {test_ndcg:.4f} | Hit@10: {test_hit:.4f}")

    history["test_ndcg"] = test_ndcg
    history["test_hit"] = test_hit

    results_path = Path(config["paths"]["results_dir"]) / f"{run_name}_results.json"
    results: Dict[str, Any] = {}
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}
    results.update({
        f"test_ndcg_{strategy_name}": test_ndcg,
        f"test_hit_{strategy_name}": test_hit,
        "history": {
            "train_losses": history["train_losses"],
            "val_epochs": history["val_epochs"],
            "val_ndcg": history["val_ndcg"],
            "val_hit": history["val_hit"],
        },
    })
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    plot_path = str(Path(config["paths"]["results_dir"]) / f"{run_name}_training_curves.png")
    plot_training_curves(
        history, plot_path,
        title=f"QAT ({strategy_name.upper()}) Training — {run_name}",
    )


def apply_adaround(
    model: nn.Module,
    train_loader: DataLoader,
    dataset: List,
    args: Args,
    fp32_checkpoint: str,
    adaround_config: Dict[str, Any],
    checkpoint_dir: str,
    save_name: str = "sasrec_adaround.pth",
    logger=None,
    config: Optional[Dict[str, Any]] = None,
):
    """Apply AdaRound PTQ."""
    print("Starting AdaRound PTQ...")
    fp32_path = os.path.join('./', fp32_checkpoint)
    if not os.path.exists(fp32_path):
        raise FileNotFoundError(f"FP32 checkpoint not found: {fp32_path}")

    print(f"Loading FP32 checkpoint: {fp32_checkpoint}")
    ckpt = torch.load(fp32_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.prepare_quant("adaround", adaround_config)

    if logger is not None and hasattr(model, "quant_strategy") and hasattr(model.quant_strategy, "set_logger"):
        model.quant_strategy.set_logger(logger)

    print("Running AdaRound calibration...")
    model.calibrate(train_loader)

    print("Validating AdaRound model on validation split...")
    ndcg_val, ht_val = evaluate(model, dataset, args, logger=logger, epoch=1)
    print(f"[AdaRound][Val] NDCG@10: {ndcg_val:.4f} | Hit@10: {ht_val:.4f}")

    print("Running final test evaluation for AdaRound model (user_test split)...")
    ndcg_test, ht_test = evaluate_test(model, dataset, args, logger=None, epoch=0)
    print(f"[AdaRound][Test] NDCG@10: {ndcg_test:.4f} | Hit@10: {ht_test:.4f}")

    save_checkpoint(
        model,
        None,
        0,
        checkpoint_dir,
        save_name,
        metrics={
            "ndcg_val": ndcg_val,
            "hit_val": ht_val,
            "ndcg_test": ndcg_test,
            "hit_test": ht_test,
            "config": adaround_config,
        },
    )
    print(f"AdaRound done. Model saved as {save_name}")

    if config is not None:
        run_name = config["experiment"]["run_name"]
        results_dir = Path(config["paths"]["results_dir"])
        results_path = results_dir / f"{run_name}_results.json"
        results: Dict[str, Any] = {}
        if results_path.exists():
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
            except Exception:
                results = {}
        results.update({
            "ndcg_val": ndcg_val,
            "hit_val": ht_val,
            "ndcg_test": ndcg_test,
            "hit_test": ht_test,
        })
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        plot_path = str(results_dir / f"{run_name}_metrics.png")
        plot_adaround_metrics(
            ndcg_val, ht_val, ndcg_test, ht_test,
            save_path=plot_path,
            title=f"AdaRound Metrics — {run_name}",
        )
