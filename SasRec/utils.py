"""Utility helpers for configuration management, logging, and training."""

from __future__ import annotations

import os
import logging
import random
import warnings
from collections.abc import Mapping

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

LOGGER = logging.getLogger("quant_experiments")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a basic logging setup shared by training and evaluation."""
    if logging.getLogger().handlers:
        # Respect existing handlers (e.g. when running inside notebooks).
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def deep_update(target: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``target`` and return the result."""
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            target[key] = deep_update(dict(target[key]), value)
        else:
            target[key] = value
    return target


def _resolve_config_path(base_path: Path, candidate: Optional[str]) -> Optional[Path]:
    if not candidate:
        return None
    candidate_path = Path(candidate)
    if not candidate_path.is_absolute():
        candidate_path = (base_path.parent / candidate_path).resolve()
    return candidate_path


def load_yaml_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return data


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file, supporting ``base_config`` and ``model_config`` keys."""

    def _load(path: Path, stack: Optional[List[Path]] = None) -> Dict[str, Any]:
        stack = stack or []
        if path in stack:
            raise RuntimeError(f"Circular config dependency detected: {stack + [path]}")
        data = load_yaml_file(path)
        merged: Dict[str, Any] = {}

        base_cfg = _resolve_config_path(path, data.get("base_config"))
        if base_cfg:
            merged = deep_update(merged, _load(base_cfg, stack + [path]))

        model_cfg = _resolve_config_path(path, data.get("model_config"))
        if model_cfg:
            merged = deep_update(merged, _load(model_cfg, stack + [path]))

        current = {k: v for k, v in data.items() if k not in {"base_config", "model_config"}}
        merged = deep_update(merged, current)

        merged.setdefault("_metadata", {})
        merged["_metadata"]["loaded_from"] = str(path)
        return merged

    config = _load(Path(config_path).resolve())
    return config


def ensure_dir(path: str | Path, create: bool = True) -> Path:
    """Ensure a directory exists and return it as ``Path``."""
    path_obj = Path(path)
    if create:
        path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj



def set_random_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_clearml_task(logging_cfg: Mapping[str, Any], full_config: Mapping[str, Any]):
    """Initialize a ClearML task if requested."""
    if os.getenv("CLEARML_DISABLE", "0") == "1":
        return None
    backend = logging_cfg.get("backend")
    if backend != "clearml":
        return None
    try:
        from clearml import Task
    except ImportError:
        warnings.warn(
            "ClearML logging requested but the 'clearml' package is not installed.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    clearml_cfg = logging_cfg.get("clearml", {})
    project = clearml_cfg.get("project_name", "lstm-int8-quant")
    raw_task_name = clearml_cfg.get("task_name")
    if raw_task_name is None or str(raw_task_name).lower() in {"none", "null", ""}:
        loaded_from = full_config.get("_metadata", {}).get("loaded_from")
        if loaded_from:
            task_name = Path(loaded_from).stem
        else:
            # Fallbacks: experiment.run_name or experiment.name
            exp_cfg = full_config.get("experiment", {}) or {}
            task_name = exp_cfg.get("run_name") or exp_cfg.get("name") or "quant_experiment"
    else:
        task_name = raw_task_name

    tags = clearml_cfg.get("tags", [])

    task = Task.init(project_name=project, task_name=task_name, tags=tags)
    try:
        task.connect(full_config, name="config")
    except TypeError:
        try:
            task.connect(full_config)
        except Exception:
            pass
    return task


def ndcg_k(actual, predicted, k=10):
    """
    Computes NDCG at k.
    actual: list of relevant items (usually just one [item_id])
    predicted: list of predicted items (ranked)
    """
    idcg = 1.0
    dcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            dcg += 1.0 / np.log2(i + 2)
    return dcg / idcg

def hit_k(actual, predicted, k=10):
    """
    Computes Hit at k.
    """
    for p in predicted[:k]:
        if p in actual:
            return 1.0
    return 0.0
