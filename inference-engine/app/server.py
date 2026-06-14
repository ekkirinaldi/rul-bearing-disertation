"""FastAPI server: REST metadata + WebSocket streaming."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.engine import StreamSession, load_model
from app.model_registry import feature_names, list_datasets, stream_label

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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse(WEB_DIR / "favicon.ico", media_type="image/x-icon")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


async def _stream_loop(
    ws: WebSocket,
    session: StreamSession,
    state: dict,
) -> None:
    """Emit one frame per acquisition until done or cancelled."""
    while state.get("streaming") and not state.get("paused"):
        payload = session.step()
        if payload is None:
            await ws.send_json({"type": "done", "t": session.t})
            state["streaming"] = False
            break
        await ws.send_json({"type": "frame", **payload})
        await asyncio.sleep(state.get("speed_ms", 50) / 1000.0)


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    session: StreamSession | None = None
    state: dict = {"streaming": False, "paused": False, "speed_ms": 50}
    stream_task: asyncio.Task | None = None

    async def cancel_stream() -> None:
        nonlocal stream_task
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

            elif action == "set_speed":
                state["speed_ms"] = max(10, int(msg.get("speed_ms", state["speed_ms"])))
                await ws.send_json({"type": "speed", "speed_ms": state["speed_ms"]})

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
