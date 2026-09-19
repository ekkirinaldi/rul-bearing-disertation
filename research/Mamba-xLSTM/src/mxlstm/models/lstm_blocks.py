"""Plain ``nn.LSTM`` stack used as the A5 ablation against xLSTM.

Drops xLSTM's exponential input gate and matrix memory entirely, giving us
the standard LSTM that the §4 / Phase 4 ablation table calls for. Same
forward signature as :class:`mxlstm.models.xlstm_blocks.XLSTMStack` so it
can be hot-swapped inside :class:`MambaXLSTMNet`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PlainLSTMStack(nn.Module):
    """Stack of ``nn.LSTM`` layers with residual + LayerNorm + dropout.

    The wrapper preserves ``d_model`` across layers so the output can be
    fed into the same fusion / regression head as the xLSTM stack.
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_blocks: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.bidirectional = bidirectional
        layers: list[nn.Module] = []
        proj_layers: list[nn.Module] = []
        for _ in range(num_blocks):
            layers.append(
                nn.LSTM(
                    input_size=d_model,
                    hidden_size=d_model,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                )
            )
            proj_layers.append(
                nn.Linear(d_model * 2, d_model) if bidirectional else nn.Identity()
            )
        self.layers = nn.ModuleList(layers)
        self.projs = nn.ModuleList(proj_layers)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for lstm, proj, norm in zip(self.layers, self.projs, self.norms, strict=False):
            y, _ = lstm(out)
            y = proj(y)
            out = norm(out + self.dropout(y))
        return out
