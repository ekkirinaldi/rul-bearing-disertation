"""XJTU-SY preprocessing aligned with Liu et al. (Sensors 2026, 26, 1578).

Uses the same §3.2 feature set, ``T_o`` selection, and ISOMAP scalar HI as
:mod:`mxlstm.data.liu2026_phm` (horizontal channel only). Per §3.1.1 / §3.3.1,
the network input additionally includes a **normalized time index** column
(linear in ``[0, 1]`` over the bearing life), so :meth:`transform_signal`
returns shape ``(T, 2)`` — ``[liu_isomap_hi, time_idx]``.

Triggered by ``hi_pipeline: liu2026_xjtu`` in ``configs/data/xjtu_sy_liu2026_strict.yaml``
(or ``xjtu_sy_paper.yaml``).
"""

from __future__ import annotations

import numpy as np

from mxlstm.data.liu2026_phm import Liu2026PhmHiPipeline


class Liu2026XjtuHiPipeline(Liu2026PhmHiPipeline):
    """Same fit/transform_hi_raw as PHM Liu pipeline; signal output appends time column."""

    def transform_signal(self, signal: np.ndarray) -> np.ndarray:
        """``(T, 1, L)`` → ``(T, 2)``: smoothed ISOMAP HI + normalized acquisition index."""
        hi1 = super().transform_signal(signal).astype(np.float32)
        t = hi1.shape[0]
        if t == 0:
            return hi1
        time_col = np.linspace(0.0, 1.0, t, dtype=np.float32).reshape(-1, 1)
        return np.concatenate([hi1, time_col], axis=1)
