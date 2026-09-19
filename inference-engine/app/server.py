"""FastAPI server: REST metadata + WebSocket streaming."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.dissertation import dissertation_context
from app.engine import _IG_STREAM_EVERY, StreamSession, load_model
from app.model_registry import feature_names, list_datasets, stream_label
from app import replay_store

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title="RUL Streaming Inference Engine", version="0.1.0")


@app.on_event("startup")
async def preload_models() -> None:
    """Warm-load both dissertation models (PHM2012 + XJTU-SY)."""
    for spec in list_datasets():
        try:
            load_model(spec.key)
        except Exception as exc:  # noqa: BLE001 — log but allow partial startup
            print(f"[startup] failed to load {spec.key}: {exc}")


@app.get("/api/datasets")
def api_datasets() -> dict:
    items = []
    for spec in list_datasets():
        items.append(
            {
                "key": spec.key,
                "label": spec.label,
                "test_bearings": spec.test_bearings,
                "bearing_labels": {
                    b: stream_label(spec.key, b) for b in spec.test_bearings
                },
                "window_length": spec.window_length,
                "acquisition_interval_s": spec.acquisition_interval_s,
                "model": spec.model_name,
                "checkpoint": spec.checkpoint.name,
                "has_gt_rul": spec.has_gt_rul,
                "transfer_note": spec.transfer_note,
                "stream_mode": spec.stream_mode,
                "feature_names": feature_names(spec),
            }
        )
    return {"datasets": items}


@app.get("/api/dissertation/{key}")
def api_dissertation(key: str) -> dict:
    """Dissertation-derived context: RUL metrics + SAE→BPFx + backbone/tier."""
    try:
        return dissertation_context(key)
    except KeyError:
        return {"key": key, "metrics": None, "bpfx": None, "error": "unknown dataset"}


@app.get("/api/runs")
def api_runs() -> dict:
    """Index of saved (replayable) runs — meta + summary, no frame payloads."""
    return {"runs": replay_store.list_runs()}


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict:
    """Full saved run: meta, drops (with explanations), and every frame."""
    run = replay_store.load_run(run_id)
    if run is None:
        return {"error": "not found", "run_id": run_id}
    return run


@app.delete("/api/runs/{run_id}")
def api_run_delete(run_id: str) -> dict:
    return {"ok": replay_store.delete_run(run_id), "run_id": run_id}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse(WEB_DIR / "favicon.ico", media_type="image/x-icon")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


async def _stream_ig(ws: WebSocket, session: StreamSession) -> None:
    """Compute one throttled, reduced-step Integrated Gradients pass off the loop.

    The heavy Captum pass runs in a worker thread; the session's model lock keeps
    it from colliding with the per-acquisition prediction. Failures are swallowed
    so streaming IG never breaks the run.
    """
    try:
        expl = await asyncio.to_thread(session.compute_ig_stream)
        if expl is not None:
            await ws.send_json(
                {"type": "explanation", "ok": True, "streaming": True, "t": session.t, **expl}
            )
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — IG must never break the stream
        pass


async def _stream_loop(
    ws: WebSocket,
    session: StreamSession,
    state: dict,
) -> None:
    """Emit one frame per acquisition until done or cancelled.

    The model-bound ``step`` runs off the event loop (``to_thread``) so the loop
    stays responsive to pause/seek/speed while inference runs. When ``auto_ig`` is
    on, a throttled background IG pass is fired every ``_IG_STREAM_EVERY``
    acquisitions and never more than one at a time.
    """
    try:
        while state.get("streaming") and not state.get("paused"):
            payload = await asyncio.to_thread(session.step)
            if payload is None:
                await ws.send_json({"type": "done", "t": session.t})
                state["streaming"] = False
                break
            await ws.send_json({"type": "frame", **payload})
            if (
                state.get("auto_ig")
                and not payload.get("warmup")
                and session.t % _IG_STREAM_EVERY == 0
            ):
                prev = state.get("ig_task")
                if prev is None or prev.done():
                    state["ig_task"] = asyncio.create_task(_stream_ig(ws, session))
            await asyncio.sleep(state.get("speed_ms", 50) / 1000.0)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface stream failures to the client
        state["streaming"] = False
        await ws.send_json({"type": "error", "message": str(exc)})


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    session: StreamSession | None = None
    state: dict = {
        "streaming": False,
        "paused": False,
        "speed_ms": 50,
        "auto_ig": True,
        "ig_task": None,
    }
    stream_task: asyncio.Task | None = None

    async def cancel_stream() -> None:
        nonlocal stream_task
        ig_task = state.get("ig_task")
        if ig_task and not ig_task.done():
            ig_task.cancel()
            try:
                await ig_task
            except asyncio.CancelledError:
                pass
        state["ig_task"] = None
        if stream_task and not stream_task.done():
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
        stream_task = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            action = msg.get("action", "start")

            if action in ("start", "stream"):
                await cancel_stream()
                dataset = str(msg.get("dataset", "phm2012"))
                bearing = str(msg.get("bearing", "1_3"))
                state["speed_ms"] = max(10, int(msg.get("speed_ms", state["speed_ms"])))
                session = StreamSession(dataset_key=dataset, bearing_id=bearing)
                state["streaming"] = True
                state["paused"] = False
                await ws.send_json(
                    {
                        "type": "started",
                        "n_total": session.n_total,
                        "window_length": session.window_length,
                        "interval_s": session.interval_s,
                        "device": str(session.loaded.device),
                        "model": session.spec.model_name,
                        "checkpoint": session.spec.checkpoint.name,
                        "has_gt_rul": session.spec.has_gt_rul,
                        "feature_names": feature_names(session.spec),
                    }
                )
                stream_task = asyncio.create_task(_stream_loop(ws, session, state))

            elif action == "pause":
                state["paused"] = True
                await ws.send_json({"type": "paused"})

            elif action == "resume":
                if session is not None and state.get("streaming"):
                    state["paused"] = False
                    await ws.send_json({"type": "resumed"})
                    if stream_task is None or stream_task.done():
                        stream_task = asyncio.create_task(_stream_loop(ws, session, state))
                else:
                    await ws.send_json({"type": "resumed"})

            elif action == "reset":
                await cancel_stream()
                if session is not None:
                    session.reset()
                    state["streaming"] = False
                    state["paused"] = False
                    await ws.send_json({"type": "reset"})

            elif action == "step":
                await cancel_stream()
                if session is None:
                    dataset = str(msg.get("dataset", "phm2012"))
                    bearing = str(msg.get("bearing", "1_3"))
                    session = StreamSession(dataset_key=dataset, bearing_id=bearing)
                state["streaming"] = False
                state["paused"] = False
                payload = session.step()
                if payload is None:
                    await ws.send_json({"type": "done", "t": session.t})
                else:
                    await ws.send_json({"type": "frame", **payload})

            elif action == "seek":
                await cancel_stream()
                if session is not None:
                    t = int(msg.get("t", 0))
                    state["streaming"] = False
                    frame = session.seek(t)
                    await ws.send_json({"type": "seek", "t": session.t, "frame": frame})

            elif action == "explain":
                if session is not None:
                    try:
                        explanation = await asyncio.to_thread(session.explain_current)
                        if explanation is None:
                            await ws.send_json(
                                {"type": "explanation", "ok": False, "reason": "warming up"}
                            )
                        else:
                            await ws.send_json(
                                {"type": "explanation", "ok": True, "t": session.t, **explanation}
                            )
                    except Exception as exc:  # noqa: BLE001
                        await ws.send_json(
                            {"type": "explanation", "ok": False, "reason": str(exc)}
                        )

            elif action == "record":
                # Headless full run-to-failure capture: stream the whole bearing
                # as fast as the model allows, persist every value, then locate
                # and explain the significant RUL drops. Progress is emitted from
                # the worker thread back onto the event loop.
                await cancel_stream()
                state["streaming"] = False
                state["paused"] = False
                dataset = str(msg.get("dataset", "phm2012"))
                bearing = str(msg.get("bearing", "1_3"))
                await ws.send_json(
                    {"type": "record_started", "dataset": dataset, "bearing": bearing}
                )
                loop = asyncio.get_running_loop()
                last_emit = {"t": -1}

                def progress(t: int, n_total: int) -> None:
                    step = max(1, n_total // 60)
                    if t - last_emit["t"] >= step or t >= n_total:
                        last_emit["t"] = t
                        asyncio.run_coroutine_threadsafe(
                            ws.send_json(
                                {"type": "record_progress", "t": t, "n_total": n_total}
                            ),
                            loop,
                        )

                try:
                    run = await asyncio.to_thread(
                        replay_store.record_run, dataset, bearing, progress_cb=progress
                    )
                    await ws.send_json(
                        {
                            "type": "record_saved",
                            "run_id": run["run_id"],
                            "meta": run["meta"],
                            "summary": run["summary"],
                            "drops": run["drops"],
                        }
                    )
                except Exception as exc:  # noqa: BLE001 — surface record failures
                    await ws.send_json({"type": "record_error", "message": str(exc)})

            elif action == "record_all":
                # Sequentially record every dataset+bearing into the run library,
                # forwarding per-item and per-acquisition progress to the client.
                await cancel_stream()
                state["streaming"] = False
                state["paused"] = False
                await ws.send_json({"type": "record_all_started"})
                loop = asyncio.get_running_loop()
                last_frame = {"t": -1}

                def item_cb(done: int, total: int, ds: str, b: str, phase: str) -> None:
                    last_frame["t"] = -1
                    asyncio.run_coroutine_threadsafe(
                        ws.send_json(
                            {
                                "type": "record_all_item",
                                "done": done,
                                "total": total,
                                "dataset": ds,
                                "bearing": b,
                                "phase": phase,
                            }
                        ),
                        loop,
                    )

                def frame_cb(idx: int, total: int, ds: str, b: str, t: int, n: int) -> None:
                    step = max(1, n // 50)
                    if t - last_frame["t"] >= step or t >= n:
                        last_frame["t"] = t
                        asyncio.run_coroutine_threadsafe(
                            ws.send_json(
                                {
                                    "type": "record_all_frame",
                                    "idx": idx,
                                    "total": total,
                                    "dataset": ds,
                                    "bearing": b,
                                    "t": t,
                                    "n": n,
                                }
                            ),
                            loop,
                        )

                try:
                    results = await asyncio.to_thread(
                        replay_store.record_all,
                        skip_existing=bool(msg.get("skip_existing", True)),
                        item_cb=item_cb,
                        frame_cb=frame_cb,
                    )
                    await ws.send_json({"type": "record_all_done", "results": results})
                except Exception as exc:  # noqa: BLE001
                    await ws.send_json({"type": "record_all_error", "message": str(exc)})

            elif action == "set_speed":
                state["speed_ms"] = max(10, int(msg.get("speed_ms", state["speed_ms"])))
                await ws.send_json({"type": "speed", "speed_ms": state["speed_ms"]})

            elif action == "set_auto_ig":
                state["auto_ig"] = bool(msg.get("on", True))
                await ws.send_json({"type": "auto_ig", "on": state["auto_ig"]})

            elif action == "stop":
                await cancel_stream()
                state["streaming"] = False
                await ws.send_json({"type": "stopped"})

    except WebSocketDisconnect:
        await cancel_stream()
    except Exception as exc:
        await cancel_stream()
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
