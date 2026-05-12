"""Lightning callbacks bundle: best-checkpoint + early stop + LR monitor."""

from __future__ import annotations

import os
from pathlib import Path

from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar


class EpochProgressPrinter(Callback):
    """Writes one flushed line per validation epoch so long runs stay visible under nohup/tee."""

    def __init__(self, metric_keys: tuple[str, ...] | None = None) -> None:
        super().__init__()
        self._metric_keys = metric_keys or (
            "train/loss_epoch",
            "train/loss",
            "val/loss",
            "val/rmse",
            "val/mae",
            "val/r2",
            "val/phm_score",
        )

    @staticmethod
    def _scalar_str(value) -> str:
        if value is None:
            return ""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            try:
                return f"{float(value.item()):.6f}"
            except (ValueError, TypeError):
                return str(value.item())
        if isinstance(value, (float, int)):
            return f"{float(value):.6f}"
        return str(value)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: ARG002
        if getattr(trainer, "sanity_checking", False):
            return
        prefix = os.environ.get("MXLSTM_PROGRESS_PREFIX", "").strip()
        epoch = trainer.current_epoch + 1
        max_epochs = getattr(trainer, "max_epochs", None) or 0
        m = getattr(trainer, "callback_metrics", {}) or {}
        parts: list[str] = []
        for key in self._metric_keys:
            if key not in m:
                continue
            parts.append(f"{key}={self._scalar_str(m[key])}")
        blob = " ".join(parts) if parts else "(metrics pending — check csv_logs)"
        head = f"{prefix} " if prefix else ""
        msg = f"{head}epoch {epoch}/{max_epochs} | {blob}"
        print(msg, flush=True)


def make_callbacks(
    *,
    checkpoint_dir: str | Path,
    monitor: str = "val/rmse",
    mode: str = "min",
    patience: int = 20,
    save_top_k: int = 1,
    use_rich_progress: bool = True,
    epoch_progress_lines: bool = True,
    checkpointing: bool = True,
) -> list:
    callbacks: list = []
    if checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="{epoch:03d}",
                monitor=monitor,
                mode=mode,
                save_top_k=save_top_k,
                save_last=True,
                auto_insert_metric_name=False,
            )
        )
    callbacks.extend(
        [
            EarlyStopping(monitor=monitor, mode=mode, patience=patience, verbose=False),
            LearningRateMonitor(logging_interval="epoch"),
        ]
    )
    if epoch_progress_lines:
        callbacks.append(EpochProgressPrinter())
    if use_rich_progress:
        callbacks.append(RichProgressBar())
    return callbacks
