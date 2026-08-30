"""Spec -> Presentation.

The public surface of the package: load a YAML deck description, validate
it, and render it to a .pptx.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pptx import Presentation
from pptx.util import Inches

from .blocks import BLOCKS
from .layout import measure_stack
from .slides import LAYOUTS
from .theme import Grid


# --------------------------------------------------------------------------
# Spec loading
# --------------------------------------------------------------------------
def load_spec(path: str | os.PathLike) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)

    spec.setdefault("meta", {})
    meta = spec["meta"]
    meta.setdefault("institution", "Institut Teknologi Bandung")
    meta.setdefault("footer", "")
    meta.setdefault("author", "")
    meta.setdefault("nim", "")

    # Resolve asset paths relative to the spec file.
    base = path.parent
    for slide in spec.get("slides", []):
        if slide.get("logo"):
            slide["logo"] = _resolve(slide["logo"], base)
        _resolve_images(slide.get("blocks", []), base)
    return spec


def _resolve(candidate: str, base: Path) -> str:
    path = Path(candidate)
    return str(path if path.is_absolute() else (base / path).resolve())


def _resolve_images(blocks: list[dict], base: Path) -> None:
    for block in blocks:
        if block.get("image"):
            block["image"] = _resolve(block["image"], base)
        for key in ("blocks", "items"):
            children = block.get(key)
            if isinstance(children, list):
                nested = [c for c in children if isinstance(c, dict)]
                _resolve_images(nested, base)
                for child in nested:
                    if isinstance(child.get("blocks"), list):
                        _resolve_images(child["blocks"], base)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def lint_spec(spec: dict) -> list[str]:
    """Structural + overflow checks. Returns human-readable problems."""
    problems: list[str] = []
    band = Grid.CONTENT_W

    for index, slide in enumerate(spec.get("slides", []), start=1):
        label = f"slide {index} ({slide.get('title', slide.get('layout', '?'))!r})"
        layout = slide.get("layout", "content")
        if layout not in LAYOUTS:
            problems.append(f"{label}: unknown layout {layout!r}")
            continue
        if layout in ("content", "section") and not slide.get("title"):
            problems.append(f"{label}: missing title")

        blocks = slide.get("blocks", [])
        for block in _walk(blocks):
            if block.get("type") not in BLOCKS:
                problems.append(f"{label}: unknown block type {block.get('type')!r}")

        if layout != "content" or not blocks:
            continue
        top = slide.get("top", Grid.BAND_TOP)
        bottom = slide.get("bottom", Grid.BAND_BOTTOM)
        try:
            needed = measure_stack(blocks, band)
        except KeyError as exc:
            problems.append(f"{label}: {exc}")
            continue
        available = bottom - top
        if needed > available + 0.02:
            problems.append(
                f"{label}: content overflows by {needed - available:.2f} in "
                f"(needs {needed:.2f}, band is {available:.2f})"
            )
    return problems


def _walk(blocks: list[dict]):
    for block in blocks:
        if not isinstance(block, dict):
            continue
        yield block
        if isinstance(block.get("blocks"), list):
            yield from _walk(block["blocks"])
        if isinstance(block.get("items"), list):
            for item in block["items"]:
                if isinstance(item, dict) and isinstance(item.get("blocks"), list):
                    yield from _walk(item["blocks"])


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def build_deck(spec: dict, out_path: str | os.PathLike) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(Grid.SLIDE_W)
    prs.slide_height = Inches(Grid.SLIDE_H)

    meta = spec["meta"]
    page = 0
    for slide_spec in spec.get("slides", []):
        layout = slide_spec.get("layout", "content")
        page += 1
        if layout == "cover":
            LAYOUTS["cover"](prs, slide_spec, meta)
        elif layout == "closing":
            LAYOUTS["closing"](prs, slide_spec, meta)
        elif layout == "section":
            LAYOUTS["section"](prs, slide_spec, meta, page)
        else:
            LAYOUTS["content"](prs, slide_spec, meta, page)

    core = prs.core_properties
    core.title = spec.get("meta", {}).get("deck_title", "Presentasi Disertasi")
    core.author = meta.get("author", "")
    core.subject = meta.get("program", "")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out
