"""Cross-condition generalization: train on one condition, test on another.

Example::

    python scripts/run_generalization.py \
        --data configs/data/phm2012.yaml \
        --model configs/model/mamba_xlstm_net.yaml \
        --train configs/train/default.yaml \
        --train-condition 1 --test-condition 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "src"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from omegaconf import OmegaConf

from mxlstm.utils.config import load_configs


_PHM_BEARINGS = {
    1: ["1_1", "1_2", "1_3", "1_4", "1_5", "1_6", "1_7"],
    2: ["2_1", "2_2", "2_3", "2_4", "2_5", "2_6", "2_7"],
    3: ["3_1", "3_2", "3_3"],
}
_XJTU_BEARINGS = {
    1: ["1_1", "1_2", "1_3", "1_4", "1_5"],
    2: ["2_1", "2_2", "2_3", "2_4", "2_5"],
    3: ["3_1", "3_2", "3_3", "3_4", "3_5"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--train-condition", type=int, required=True)
    p.add_argument("--test-condition", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_configs([args.data, args.model, args.train])

    if cfg.data.dataset == "phm2012":
        bearings = _PHM_BEARINGS
    elif cfg.data.dataset in ("xjtusy", "xjtu_sy"):
        bearings = _XJTU_BEARINGS
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset}")

    # Override the splits in-memory then call train.py's main().
    cfg.data.train_bearings = bearings[args.train_condition][:-1]    # leave last bearing for val
    cfg.data.val_bearings = [bearings[args.train_condition][-1]]
    cfg.data.test_bearings = bearings[args.test_condition]

    # Persist the temp config and call train.py directly.
    tmp = Path("results/_gen_tmp.yaml")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, tmp)

    import subprocess
    cmd = [
        sys.executable, "scripts/train.py",
        "--data", str(tmp),
        "--model", str(args.model),
        "--train", str(args.train),
        "--seed", str(args.seed),
        "--run-id", f"gen_c{args.train_condition}_to_c{args.test_condition}_s{args.seed}",
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
