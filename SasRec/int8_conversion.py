"""
Convert the best QAT model (QDrop 8-bit) to real INT8 via PyTorch dynamic
quantization, then benchmark quality (NDCG@10) and CPU speed.

Run from SasRec/ directory:
    python int8_conversion.py

The script:
  1. Loads data (MovieLens-1M).
  2. Restores clean FP32 weights from the QDrop checkpoint (fake-quant wrappers
     are stripped; only weight/bias tensors are loaded, strict=False).
  3. Applies torch.quantization.quantize_dynamic to all nn.Linear layers
     → real INT8 weight storage, INT8 GEMM on CPU.
  4. Measures NDCG@10 and Hit@10 on both val and test splits.
  5. Benchmarks CPU latency / throughput.
  6. Prints a before/after comparison table and saves results JSON + bar chart.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

# Make sure we're on CPU (INT8 backend)
DEVICE = torch.device("cpu")


# ── Imports from SasRec project ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.dataloader import create_dataloaders
from models.quantization import QuantSASRec
from utils import load_config, ndcg_k, hit_k


# ── Config / args helper ──────────────────────────────────────────────────────
class Args:
    def __init__(self, config: Dict[str, Any]):
        self.hidden_units = config["model"]["hidden_units"]
        self.num_blocks   = config["model"]["num_blocks"]
        self.num_heads    = config["model"]["num_heads"]
        self.dropout_rate = config["model"]["dropout_rate"]
        self.maxlen       = config["model"]["maxlen"]
        self.norm_first   = config["model"].get("norm_first", False)
        self.device       = DEVICE


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data(config_path: str = "configs/base.yaml"):
    config = load_config(config_path)
    config["experiment"]["device"] = "cpu"
    config["training"]["num_workers"] = 0   # avoid fork issues when called from notebook

    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        config, seed=config["experiment"].get("seed", 42)
    )
    return config, train_loader, val_loader, test_loader, dataset


# ── Restore clean FP32 model from a QAT checkpoint ───────────────────────────
def load_fp32_from_ckpt(
    ckpt_path: str | Path,
    usernum: int,
    itemnum: int,
    args: Args,
) -> QuantSASRec:
    """
    Instantiate QuantSASRec (no fake-quant attached) and load weights from a
    QAT checkpoint with strict=False so quantizer-specific buffers are ignored.
    Result: plain float model with the trained weights.
    """
    model = QuantSASRec(usernum, itemnum, args)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"  [load] missing keys (expected — quantizer buffers): {len(missing)}")
    if unexpected:
        print(f"  [load] unexpected keys: {len(unexpected)}")

    model.eval().to(DEVICE)
    return model


# ── Apply dynamic INT8 quantization ──────────────────────────────────────────
def _select_backend() -> str:
    supported = torch.backends.quantized.supported_engines
    for backend in ("qnnpack", "x86", "fbgemm"):
        if backend in supported:
            return backend
    raise RuntimeError(f"No supported quantization backend found. Available: {supported}")


def apply_dynamic_int8(fp32_model: nn.Module) -> nn.Module:
    """
    Replace all nn.Linear layers with dynamically-quantized INT8 versions.
    Weights are stored as INT8; activations are quantized on-the-fly per batch.
    """
    backend = _select_backend()
    torch.backends.quantized.engine = backend
    print(f"  Using quantization backend: {backend}")

    int8_model = copy.deepcopy(fp32_model)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        int8_model = torch.quantization.quantize_dynamic(
            int8_model,
            qconfig_spec={nn.Linear},
            dtype=torch.qint8,
        )
    int8_model.eval().to(DEVICE)
    return int8_model


# ── Evaluation helpers ────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_val(
    model: nn.Module,
    dataset: List,
    args: Args,
    desc: str = "Val",
) -> Tuple[float, float]:
    model.eval()
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    ndcg_sum = hit_sum = valid_n = 0.0
    for u in tqdm(range(1, usernum + 1), desc=desc, leave=False):
        if len(user_train[u]) < 1 or len(user_valid[u]) < 1:
            continue
        seq = user_train[u]
        target = user_valid[u][0]

        if len(seq) >= args.maxlen:
            seq = seq[-args.maxlen:]
        else:
            seq = [0] * (args.maxlen - len(seq)) + seq

        log_seqs    = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        item_indices = list(range(1, itemnum + 1))
        preds        = -model.predict(None, log_seqs, item_indices)[0]
        _, idx       = torch.topk(preds, 10, largest=False)
        rank_list    = [item_indices[i] for i in idx.cpu().numpy()]

        ndcg_sum += ndcg_k([target], rank_list, k=10)
        hit_sum  += hit_k([target],  rank_list, k=10)
        valid_n  += 1

    return (ndcg_sum / valid_n, hit_sum / valid_n) if valid_n else (0.0, 0.0)


@torch.no_grad()
def evaluate_test(
    model: nn.Module,
    dataset: List,
    args: Args,
    desc: str = "Test",
) -> Tuple[float, float]:
    model.eval()
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    ndcg_sum = hit_sum = valid_n = 0.0
    for u in tqdm(range(1, usernum + 1), desc=desc, leave=False):
        if len(user_train[u]) < 1 or len(user_test[u]) < 1:
            continue
        seq    = user_train[u]
        target = user_test[u][0]

        if len(seq) >= args.maxlen:
            seq = seq[-args.maxlen:]
        else:
            seq = [0] * (args.maxlen - len(seq)) + seq

        log_seqs     = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        item_indices = list(range(1, itemnum + 1))
        preds        = -model.predict(None, log_seqs, item_indices)[0]
        _, idx        = torch.topk(preds, 10, largest=False)
        rank_list     = [item_indices[i] for i in idx.cpu().numpy()]

        ndcg_sum += ndcg_k([target], rank_list, k=10)
        hit_sum  += hit_k([target],  rank_list, k=10)
        valid_n  += 1

    return (ndcg_sum / valid_n, hit_sum / valid_n) if valid_n else (0.0, 0.0)


# ── CPU speed benchmark ───────────────────────────────────────────────────────
@torch.no_grad()
def benchmark_cpu(
    model: nn.Module,
    train_loader,
    warmup: int = 10,
    iters: int = 100,
    desc: str = "Benchmark",
) -> Dict[str, float]:
    model.eval().to(DEVICE)

    # collect batches
    batches = []
    for u, seq, pos, neg in train_loader:
        batches.append((u.to(DEVICE), seq.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)))
        if len(batches) >= iters:
            break

    # warm-up
    for u, seq, pos, neg in batches[:warmup]:
        model(u, seq, pos, neg)

    times: List[float] = []
    for u, seq, pos, neg in tqdm(batches, desc=desc, leave=False):
        t0 = time.perf_counter()
        model(u, seq, pos, neg)
        times.append(time.perf_counter() - t0)

    avg_ms    = float(np.mean(times)) * 1000
    median_ms = float(np.median(times)) * 1000
    throughput = 1.0 / (float(np.mean(times)) + 1e-12)

    return {
        "avg_latency_ms":    avg_ms,
        "median_latency_ms": median_ms,
        "throughput_batches_per_sec": throughput,
    }


# ── Model size ────────────────────────────────────────────────────────────────
def count_params_mb(model: nn.Module) -> float:
    total_bytes = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    total_bytes += sum(
        b.nelement() * b.element_size() for b in model.buffers()
    )
    return total_bytes / (1024 ** 2)


# ── Plot comparison ───────────────────────────────────────────────────────────
def plot_int8_comparison(results: Dict[str, Any], save_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    fp32 = results["fp32"]
    int8 = results["int8"]

    metrics = {
        "NDCG@10\n(val)":       (fp32["ndcg_val"],  int8["ndcg_val"]),
        "Hit@10\n(val)":        (fp32["hit_val"],   int8["hit_val"]),
        "NDCG@10\n(test)":      (fp32["ndcg_test"], int8["ndcg_test"]),
        "Hit@10\n(test)":       (fp32["hit_test"],  int8["hit_test"]),
    }
    speed_metrics = {
        "Avg latency\n(ms)":  (fp32["avg_latency_ms"],    int8["avg_latency_ms"]),
        "Throughput\n(b/s)":  (fp32["throughput"],        int8["throughput"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "QDrop (8-bit QAT) → Real INT8 (dynamic quantization)\nSASRec · MovieLens-1M · CPU",
        fontsize=12, fontweight="bold",
    )

    # Quality
    ax = axes[0]
    labels = list(metrics)
    fp32_v = [metrics[l][0] for l in labels]
    int8_v = [metrics[l][1] for l in labels]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, fp32_v, w, label="FP32 weights (QDrop)", color="#4C72B0", alpha=0.85)
    ax.bar(x + w / 2, int8_v, w, label="INT8 weights (dynamic)", color="#C44E52", alpha=0.85)
    for xi, (fv, iv) in zip(x, zip(fp32_v, int8_v)):
        ax.text(xi - w / 2, fv + max(fp32_v) * 0.012, f"{fv:.4f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, iv + max(int8_v) * 0.012, f"{iv:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Metric value")
    ax.set_title("Quality: FP32 vs INT8")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(fp32_v + int8_v) * 1.25)

    # Speed
    ax = axes[1]
    slabels  = list(speed_metrics)
    fp32_sv  = [speed_metrics[l][0] for l in slabels]
    int8_sv  = [speed_metrics[l][1] for l in slabels]
    x = np.arange(len(slabels))
    ax.bar(x - w / 2, fp32_sv, w, label="FP32", color="#4C72B0", alpha=0.85)
    ax.bar(x + w / 2, int8_sv, w, label="INT8", color="#C44E52", alpha=0.85)
    for xi, (fv, iv) in zip(x, zip(fp32_sv, int8_sv)):
        ax.text(xi - w / 2, fv + max(fp32_sv + int8_sv) * 0.012, f"{fv:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, iv + max(fp32_sv + int8_sv) * 0.012, f"{iv:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(slabels, fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title("CPU Speed: FP32 vs INT8")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("SASRec  INT8 conversion  (QDrop → dynamic INT8)")
    print("=" * 60)

    # 1. Data
    print("\n[1/5] Loading data...")
    config, train_loader, val_loader, test_loader, dataset = load_data()
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    args = Args(config)
    print(f"  usernum={usernum}, itemnum={itemnum}")

    # 2. Load FP32 model from QDrop checkpoint
    ckpt_path = Path("checkpoints/sasrec_runs/sasrec_qdrop/sasrec_qdrop.pth")
    print(f"\n[2/5] Restoring FP32 weights from: {ckpt_path}")
    fp32_model = load_fp32_from_ckpt(ckpt_path, usernum, itemnum, args)
    fp32_size  = count_params_mb(fp32_model)
    print(f"  FP32 model param size: {fp32_size:.2f} MB")

    # 3. Convert to INT8
    print("\n[3/5] Applying dynamic INT8 quantization (nn.Linear → qint8)...")
    int8_model = apply_dynamic_int8(fp32_model)
    int8_size  = count_params_mb(int8_model)
    print(f"  INT8 model param size: {int8_size:.2f} MB")
    print(f"  Size reduction: {(1 - int8_size / fp32_size) * 100:.1f}%")

    # 4. Evaluate quality
    print("\n[4/5] Evaluating quality on val and test splits...")
    fp32_ndcg_val,  fp32_hit_val  = evaluate_val(fp32_model,  dataset, args, desc="FP32 Val")
    fp32_ndcg_test, fp32_hit_test = evaluate_test(fp32_model, dataset, args, desc="FP32 Test")
    int8_ndcg_val,  int8_hit_val  = evaluate_val(int8_model,  dataset, args, desc="INT8 Val")
    int8_ndcg_test, int8_hit_test = evaluate_test(int8_model, dataset, args, desc="INT8 Test")

    # 5. Benchmark CPU speed
    print("\n[5/5] CPU speed benchmark...")
    fp32_speed = benchmark_cpu(fp32_model, train_loader, desc="FP32 speed")
    int8_speed = benchmark_cpu(int8_model, train_loader, desc="INT8 speed")

    # ── Results ──────────────────────────────────────────────────────────────
    results = {
        "fp32": {
            "ndcg_val":   fp32_ndcg_val,
            "hit_val":    fp32_hit_val,
            "ndcg_test":  fp32_ndcg_test,
            "hit_test":   fp32_hit_test,
            "avg_latency_ms":    fp32_speed["avg_latency_ms"],
            "median_latency_ms": fp32_speed["median_latency_ms"],
            "throughput":        fp32_speed["throughput_batches_per_sec"],
            "size_mb":    fp32_size,
        },
        "int8": {
            "ndcg_val":   int8_ndcg_val,
            "hit_val":    int8_hit_val,
            "ndcg_test":  int8_ndcg_test,
            "hit_test":   int8_hit_test,
            "avg_latency_ms":    int8_speed["avg_latency_ms"],
            "median_latency_ms": int8_speed["median_latency_ms"],
            "throughput":        int8_speed["throughput_batches_per_sec"],
            "size_mb":    int8_size,
        },
    }

    out_json = Path("results/sasrec_int8_comparison.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    plot_int8_comparison(results, Path("results/sasrec_int8_comparison.png"))

    # ── Print table ───────────────────────────────────────────────────────────
    fp32 = results["fp32"]
    int8 = results["int8"]

    def delta(new, old):
        return f"  ({(new - old) / max(abs(old), 1e-9) * 100:+.1f}%)"

    print("\n" + "=" * 60)
    print("  QUALITY COMPARISON  (QDrop weights)")
    print("=" * 60)
    print(f"  {'Metric':<22}  {'FP32':>10}  {'INT8':>10}  {'Delta':>10}")
    print(f"  {'-'*52}")
    for label, fv, iv in [
        ("NDCG@10 (val)",  fp32["ndcg_val"],  int8["ndcg_val"]),
        ("Hit@10  (val)",  fp32["hit_val"],   int8["hit_val"]),
        ("NDCG@10 (test)", fp32["ndcg_test"], int8["ndcg_test"]),
        ("Hit@10  (test)", fp32["hit_test"],  int8["hit_test"]),
    ]:
        print(f"  {label:<22}  {fv:>10.5f}  {iv:>10.5f}{delta(iv, fv)}")

    print()
    print("=" * 60)
    print("  SPEED COMPARISON (CPU)")
    print("=" * 60)
    print(f"  {'Metric':<26}  {'FP32':>10}  {'INT8':>10}  {'Speedup':>10}")
    print(f"  {'-'*56}")
    for label, fv, iv in [
        ("Avg latency (ms)",       fp32["avg_latency_ms"],    int8["avg_latency_ms"]),
        ("Median latency (ms)",    fp32["median_latency_ms"], int8["median_latency_ms"]),
        ("Throughput (batches/s)", fp32["throughput"],        int8["throughput"]),
    ]:
        speedup = iv / max(fv, 1e-12)
        print(f"  {label:<26}  {fv:>10.3f}  {iv:>10.3f}  {speedup:>9.2f}x")

    print()
    print(f"  Model param size:  FP32 = {fp32['size_mb']:.2f} MB  |  INT8 = {int8['size_mb']:.2f} MB  "
          f"({(1 - int8['size_mb'] / fp32['size_mb']) * 100:.1f}% smaller)")
    print("=" * 60)


if __name__ == "__main__":
    main()
