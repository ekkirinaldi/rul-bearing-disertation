"""Vertical flow layout.

`measure_stack` totals the natural height of a block list; `render_stack`
places them top-down inside a region, distributing any slack to blocks that
asked for it (``flex: true``) and honouring per-block overrides:

``h``      fixed height, skips measurement
``gap``    space *before* this block (default :data:`Grid.GAP`)
``flex``   absorb leftover vertical space (share proportional to weight)
``pin``    ``bottom`` sticks the block to the bottom of the region
"""

from __future__ import annotations

from typing import NamedTuple

from .blocks import BLOCKS, Box
from .theme import Grid

# A flex block never shrinks below this when it absorbs negative slack.
MIN_FLEX_H = 0.40


class LayoutWarning(NamedTuple):
    slide: int
    message: str


def _gap(spec: dict, first: bool) -> float:
    if first:
        return 0.0
    return spec.get("gap", Grid.GAP)


def measure_block(spec: dict, width: float) -> float:
    if "h" in spec and spec["h"] is not None:
        return float(spec["h"])
    kind = spec.get("type")
    if kind not in BLOCKS:
        raise KeyError(f"unknown block type: {kind!r}")
    return BLOCKS[kind][0](spec, width)


def measure_stack(specs: list[dict], width: float) -> float:
    total = 0.0
    for i, spec in enumerate(specs):
        total += _gap(spec, i == 0) + measure_block(spec, width)
    return total


def render_stack(slide, specs: list[dict], x: float, y: float, w: float, bottom: float) -> float:
    """Lay out `specs` from `y` downwards; returns the y after the last block."""
    if not specs:
        return y

    pinned = [s for s in specs if s.get("pin") == "bottom"]
    flowing = [s for s in specs if s.get("pin") != "bottom"]

    heights = [measure_block(s, w) for s in flowing]
    gaps = [_gap(s, i == 0) for i, s in enumerate(flowing)]

    pin_h = 0.0
    for s in pinned:
        pin_h += measure_block(s, w) + s.get("gap", Grid.GAP)

    available = bottom - y - pin_h
    natural = sum(heights) + sum(gaps)
    slack = available - natural

    # Flex blocks absorb the leftover space in both directions: a nested
    # figure without `h` measures at its 3,50 in default, so inside a fixed
    # column shorter than that the slack is negative and the figure must
    # shrink, or it overruns the blocks below it.
    flex_idx = [i for i, s in enumerate(flowing) if s.get("flex")]
    if flex_idx and slack != 0:
        weights: list[float] = []
        for i in flex_idx:
            flex = flowing[i].get("flex")
            weights.append(float(flex) if isinstance(flex, (int, float)) and not isinstance(flex, bool) else 1.0)
        total_w = sum(weights) or 1.0
        for i, weight in zip(flex_idx, weights):
            heights[i] = max(heights[i] + slack * weight / total_w, MIN_FLEX_H)

    cursor = y
    for spec, gap, height in zip(flowing, gaps, heights):
        cursor += gap
        BLOCKS[spec["type"]][1](slide, spec, Box(x, cursor, w, height))
        cursor += height

    if pinned:
        py = bottom
        for spec in reversed(pinned):
            height = measure_block(spec, w)
            py -= height
            BLOCKS[spec["type"]][1](slide, spec, Box(x, py, w, height))
            py -= spec.get("gap", Grid.GAP)
    return cursor


def overflow(specs: list[dict], width: float, available: float) -> float:
    """How much a stack exceeds `available`, in inches (0 when it fits)."""
    return max(0.0, measure_stack(specs, width) - available)
