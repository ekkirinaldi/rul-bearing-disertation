"""YAML config loader (OmegaConf-backed) with multi-file merge.

Usage::

    from mxlstm.utils.config import load_configs
    cfg = load_configs([
        "configs/data/phm2012.yaml",
        "configs/model/mamba_xlstm_net.yaml",
        "configs/train/default.yaml",
    ])
    # cfg.data.window_length, cfg.model.d_model, cfg.train.lr ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf


def load_yaml(path: str | Path) -> DictConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    cfg = OmegaConf.load(p)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Top-level config in {p} must be a mapping; got {type(cfg)}")
    return cfg


def load_configs(paths: Iterable[str | Path]) -> DictConfig:
    """Merge a list of YAML files left-to-right. Later files override earlier."""
    merged = OmegaConf.create({})
    for p in paths:
        merged = OmegaConf.merge(merged, load_yaml(p))
    assert isinstance(merged, DictConfig)
    return merged


def save_yaml(cfg: DictConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, p)
