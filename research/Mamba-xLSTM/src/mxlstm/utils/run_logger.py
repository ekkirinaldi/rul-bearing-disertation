"""Structured per-run logger with phase / step tracking.

Wraps loguru so the same call writes to:
  * stderr (rich-formatted, level INFO+)
  * ``<run_dir>/logs/run.log``  (plain text, level DEBUG+)
  * ``<run_dir>/logs/events.jsonl`` (one structured event per line)

Phases / steps:

    rl = RunLogger(run_dir, run_id="mamba_xlstm_phm2012_s42")
    with rl.phase("Data preparation"):
        with rl.step("Load PHM2012 bearings"):
            ...
        with rl.step("Extract HI"):
            rl.metric("n_features", 36)
            ...

The context managers automatically time each block and emit
``phase_start`` / ``phase_end`` / ``step_start`` / ``step_end`` events
with elapsed seconds. ``rl.metric(name, value)`` and ``rl.artefact(name,
path)`` add extra structured rows that are surfaced in the HTML report.

Call ``rl.close()`` at the end of the run (or use ``with RunLogger(...)``)
to flush handlers and write a ``summary.json`` with the per-step timing
table.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from loguru import logger as _loguru_logger


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


@dataclass
class _Event:
    ts: float
    level: str
    kind: str            # "log" | "phase_start" | "phase_end" | "step_start" | "step_end" | "metric" | "artefact"
    message: str = ""
    phase: str | None = None
    step: str | None = None
    elapsed_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------


class RunLogger:
    def __init__(self, run_dir: str | Path, *, run_id: str | None = None, console_level: str = "INFO") -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id or self.run_dir.name
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.text_path = self.logs_dir / "run.log"
        self.events_path = self.logs_dir / "events.jsonl"
        self.summary_path = self.logs_dir / "summary.json"

        # loguru is process-global; remove existing sinks once and add our two.
        _loguru_logger.remove()
        self._console_handler = _loguru_logger.add(
            sys.stderr,
            level=console_level,
            format="<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> | {message}",
            enqueue=False,
        )
        self._file_handler = _loguru_logger.add(
            self.text_path,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
            enqueue=False,
        )
        self.logger = _loguru_logger

        self._phase_stack: list[tuple[str, float]] = []
        self._step_stack: list[tuple[str, float]] = []
        self._timings: list[dict[str, Any]] = []

        # Truncate events file at start of run.
        self.events_path.write_text("")
        self._emit(_Event(ts=time.time(), level="INFO", kind="log",
                          message=f"RunLogger initialized for {self.run_id}",
                          extra={"run_dir": str(self.run_dir)}))

    # ----- low-level emit -------------------------------------------------

    def _current_phase(self) -> str | None:
        return self._phase_stack[-1][0] if self._phase_stack else None

    def _current_step(self) -> str | None:
        return self._step_stack[-1][0] if self._step_stack else None

    @staticmethod
    def _json_default(obj: Any) -> Any:
        # numpy / torch / pathlib / device / set / etc.: fall back to str().
        try:
            import numpy as np  # local import to avoid hard dep at import time

            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        return str(obj)

    def _emit(self, ev: _Event) -> None:
        with self.events_path.open("a") as f:
            f.write(json.dumps(asdict(ev), ensure_ascii=False, default=self._json_default) + "\n")

    # ----- public log methods --------------------------------------------

    def info(self, msg: str, **extra: Any) -> None:
        self.logger.info(msg)
        self._emit(_Event(ts=time.time(), level="INFO", kind="log", message=msg,
                          phase=self._current_phase(), step=self._current_step(), extra=extra))

    def warning(self, msg: str, **extra: Any) -> None:
        self.logger.warning(msg)
        self._emit(_Event(ts=time.time(), level="WARNING", kind="log", message=msg,
                          phase=self._current_phase(), step=self._current_step(), extra=extra))

    def error(self, msg: str, **extra: Any) -> None:
        self.logger.error(msg)
        self._emit(_Event(ts=time.time(), level="ERROR", kind="log", message=msg,
                          phase=self._current_phase(), step=self._current_step(), extra=extra))

    def debug(self, msg: str, **extra: Any) -> None:
        self.logger.debug(msg)
        self._emit(_Event(ts=time.time(), level="DEBUG", kind="log", message=msg,
                          phase=self._current_phase(), step=self._current_step(), extra=extra))

    def metric(self, name: str, value: Any) -> None:
        self.logger.info(f"  · {name} = {value}")
        self._emit(_Event(ts=time.time(), level="INFO", kind="metric",
                          message=f"{name}={value}",
                          phase=self._current_phase(), step=self._current_step(),
                          extra={"name": name, "value": value}))

    def artefact(self, name: str, path: str | Path) -> None:
        rel = str(Path(path))
        self.logger.info(f"  📄 {name} → {rel}")
        self._emit(_Event(ts=time.time(), level="INFO", kind="artefact",
                          message=f"{name}={rel}",
                          phase=self._current_phase(), step=self._current_step(),
                          extra={"name": name, "path": rel}))

    # ----- phase / step context managers ---------------------------------

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        t0 = time.time()
        self._phase_stack.append((name, t0))
        self.logger.info(f"▶ Phase: {name}")
        self._emit(_Event(ts=t0, level="INFO", kind="phase_start", message=name, phase=name))
        try:
            yield
        except Exception as e:
            self.error(f"Phase '{name}' failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            elapsed = time.time() - t0
            self._phase_stack.pop()
            self._timings.append({"kind": "phase", "name": name, "elapsed_s": elapsed})
            self.logger.info(f"✓ Phase '{name}' done in {elapsed:.1f}s")
            self._emit(_Event(ts=time.time(), level="INFO", kind="phase_end",
                              message=name, phase=name, elapsed_s=elapsed))

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        t0 = time.time()
        self._step_stack.append((name, t0))
        self.logger.info(f"  · Step: {name}")
        self._emit(_Event(ts=t0, level="INFO", kind="step_start", message=name,
                          phase=self._current_phase(), step=name))
        try:
            yield
        except Exception as e:
            self.error(f"Step '{name}' failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            elapsed = time.time() - t0
            self._step_stack.pop()
            self._timings.append({"kind": "step", "phase": self._current_phase(),
                                  "name": name, "elapsed_s": elapsed})
            self.logger.info(f"  ✓ Step '{name}' done in {elapsed:.2f}s")
            self._emit(_Event(ts=time.time(), level="INFO", kind="step_end",
                              message=name, phase=self._current_phase(), step=name,
                              elapsed_s=elapsed))

    # ----- finalize -------------------------------------------------------

    def close(self) -> None:
        self.summary_path.write_text(json.dumps({
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "timings": self._timings,
        }, indent=2, default=self._json_default))
        try:
            self.logger.remove(self._console_handler)
            self.logger.remove(self._file_handler)
        except ValueError:
            pass

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Module-level helper for quick scripts
# ---------------------------------------------------------------------------


def attach_to_existing_dir(run_dir: str | Path) -> RunLogger:
    """Attach a RunLogger to an existing directory (no truncation if files exist)."""
    return RunLogger(run_dir)
