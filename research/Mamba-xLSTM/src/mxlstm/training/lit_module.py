"""LightningModule wrapping any (B, L, F) -> (B,) regression model.

Tracks RMSE/MAE/R^2/PHM Score on train/val/test and exposes hooks to
collect per-bearing predictions for the reporting layer.
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from mxlstm.eval.metrics import (
    aggregate_metrics,
    mae,
    per_bearing_metrics,
    phm_score,
    phm_score_paper,
    r2,
    rmse,
)
from mxlstm.training.losses import monotonicity_penalty, mse_loss


class RULLitModule(pl.LightningModule):
    """Trainer for any RUL model that maps ``(B, L, F) -> (B,) in [0, 1]``."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 200,
        monotonicity_weight: float = 0.01,
        scheduler: str = "cosine",
        model_specific_loss: bool = False,
        xlstm_lr_mult: float = 1.0,
        freeze_xlstm_epochs: int = 0,
    ) -> None:
        super().__init__()
        # ``model`` is not a hyperparameter (would pickle a network); save
        # the rest so Lightning can serialize them with the checkpoint.
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self._val_outputs: list[dict[str, Any]] = []
        self._test_outputs: list[dict[str, Any]] = []

    # ----- forward / step --------------------------------------------------

    def forward(self, x: torch.Tensor, condition_ids: torch.Tensor | None = None) -> torch.Tensor:
        fwd = self.model.forward
        if "condition_ids" in inspect.signature(fwd).parameters:
            return self.model(x, condition_ids=condition_ids)
        return self.model(x)

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]],
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        x, y, rul_window, metas = batch
        cids = torch.tensor(
            [int(m.get("condition", 1)) for m in metas],
            device=x.device,
            dtype=torch.long,
        )
        pred = self(x, condition_ids=cids)
        if (
            bool(getattr(self.hparams, "model_specific_loss", False))
            and stage == "train"
            and callable(getattr(self.model, "compute_loss", None))
        ):
            cl = self.model.compute_loss
            sig = inspect.signature(cl)
            params = sig.parameters
            kwargs: dict[str, Any] = {}
            if "rul_window" in params:
                kwargs["rul_window"] = rul_window
            if "condition_ids" in params:
                kwargs["condition_ids"] = cids
            loss = cl(x, y, **kwargs) if kwargs else cl(x, y)
        else:
            loss = mse_loss(pred, y)
        if self.hparams.monotonicity_weight > 0 and stage == "train":
            t_idx = torch.tensor([m["t_index"] for m in metas], device=pred.device)
            mono = monotonicity_penalty(pred, t_idx)
            loss = loss + float(self.hparams.monotonicity_weight) * mono
        return loss, pred.detach(), y.detach(), metas

    def on_train_epoch_start(self) -> None:
        """Optional: freeze mLSTM front-end for the first ``freeze_xlstm_epochs`` epochs."""
        n = int(getattr(self.hparams, "freeze_xlstm_epochs", 0) or 0)
        if n <= 0:
            return
        freeze = self.current_epoch < n
        for name, p in self.model.named_parameters():
            if any(k in name for k in ("_feat_to_d", "_xlstm_blocks", "_xlstm_norms")):
                p.requires_grad = not freeze

    def training_step(self, batch, batch_idx):
        loss, pred, y, _ = self._shared_step(batch, "train")
        bs = y.size(0)
        self.log("train/loss", loss, prog_bar=True, batch_size=bs, on_epoch=True)
        self.log("train/rmse", rmse(y, pred), prog_bar=False, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, y, metas = self._shared_step(batch, "val")
        bs = y.size(0)
        self.log("val/loss", loss, prog_bar=True, batch_size=bs)
        self._val_outputs.append({"pred": pred.cpu(), "y": y.cpu(), "metas": metas})
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, y, metas = self._shared_step(batch, "test")
        bs = y.size(0)
        self.log("test/loss", loss, prog_bar=True, batch_size=bs)
        self._test_outputs.append({"pred": pred.cpu(), "y": y.cpu(), "metas": metas})
        return loss

    # ----- epoch ends: aggregate per-bearing predictions ------------------

    @staticmethod
    def _gather_per_bearing(outs: list[dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
        per: dict[str, dict[int, float]] = defaultdict(dict)
        per_y: dict[str, dict[int, float]] = defaultdict(dict)
        for o in outs:
            preds = o["pred"].float().numpy()
            ys = o["y"].float().numpy()
            for i, m in enumerate(o["metas"]):
                bid, t = m["bearing_id"], int(m["t_index"])
                per[bid][t] = float(preds[i])
                per_y[bid][t] = float(ys[i])
        out: dict[str, dict[str, np.ndarray]] = {}
        for bid in per:
            ts = sorted(per[bid].keys())
            out[bid] = {
                "t": np.asarray(ts, dtype=np.int32),
                "pred": np.asarray([per[bid][t] for t in ts], dtype=np.float32),
                "y": np.asarray([per_y[bid][t] for t in ts], dtype=np.float32),
            }
        return out

    def _log_metrics(self, prefix: str, outs: list[dict[str, Any]]) -> None:
        if not outs:
            return
        all_pred = torch.cat([o["pred"] for o in outs]).float().numpy()
        all_y = torch.cat([o["y"] for o in outs]).float().numpy()
        self.log(f"{prefix}/rmse", float(rmse(all_y, all_pred)), prog_bar=True)
        self.log(f"{prefix}/mae", float(mae(all_y, all_pred)))
        self.log(f"{prefix}/r2", float(r2(all_y, all_pred)))
        self.log(f"{prefix}/phm_score", float(phm_score(all_y, all_pred)))
        self.log(f"{prefix}/phm_score_paper", float(phm_score_paper(all_y, all_pred)))

        per = self._gather_per_bearing(outs)
        agg = aggregate_metrics({bid: {
            "rmse": rmse(d["y"], d["pred"]),
            "mae": mae(d["y"], d["pred"]),
            "r2": r2(d["y"], d["pred"]),
            "phm_score": phm_score(d["y"], d["pred"]),
            "phm_score_paper": phm_score_paper(d["y"], d["pred"]),
        } for bid, d in per.items()})
        self.log(f"{prefix}/rmse_per_bearing", agg["rmse"])
        self.log(f"{prefix}/phm_per_bearing", agg["phm_score"])
        self.log(f"{prefix}/phm_paper_per_bearing", agg["phm_score_paper"])

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val", self._val_outputs)
        self._val_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test", self._test_outputs)
        # Keep _test_outputs for downstream evaluate.py inspection.

    # ----- optimizers ------------------------------------------------------

    def configure_optimizers(self):
        lr = float(self.hparams.lr)
        decay = float(self.hparams.weight_decay)
        mult = float(getattr(self.hparams, "xlstm_lr_mult", 1.0))
        if abs(mult - 1.0) < 1e-12:
            opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=decay)
        else:
            base: list[nn.Parameter] = []
            xlstm: list[nn.Parameter] = []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if any(k in name for k in ("_feat_to_d", "_xlstm_blocks", "_xlstm_norms")):
                    xlstm.append(p)
                else:
                    base.append(p)
            if not xlstm:
                opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=decay)
            elif not base:
                opt = torch.optim.AdamW(xlstm, lr=lr * mult, weight_decay=decay)
            else:
                opt = torch.optim.AdamW(
                    [
                        {"params": base, "lr": lr},
                        {"params": xlstm, "lr": lr * mult},
                    ],
                    weight_decay=decay,
                )
        if self.hparams.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=int(self.hparams.max_epochs),
            )
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt
