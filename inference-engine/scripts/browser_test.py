#!/usr/bin/env python3
"""Real-browser end-to-end test of the replay-library UI.

The dashboard is replay-only: the left rail is Replay Control (pick a recorded
run → it plays from acq 0), Pause/Step/Reset/Seek are the player, and the Run
Library records all runs. This test drives that flow against an already-recorded
library (it does NOT purge runs).

Assumes the server is running at $RUL_BASE (default 8813); use run_browser_test.sh.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

BASE = os.environ.get("RUL_BASE", "http://127.0.0.1:8813")


def _api(path):
    with urllib.request.urlopen(f"{BASE}{path}") as r:
        return json.load(r)


def _acq(page) -> int:
    txt = page.text_content("#mAcq") or ""
    m = re.match(r"\s*(\d+)\s*/", txt)
    return int(m.group(1)) if m else -1


def set_slider(page, t: int) -> None:
    page.evaluate(
        """(t) => {
            const s = document.getElementById('seek');
            s.value = String(t);
            s.dispatchEvent(new Event('input', { bubbles: true }));
            s.dispatchEvent(new Event('change', { bubbles: true }));
        }""",
        t,
    )


def _pick_run(dataset):
    runs = _api("/api/runs")["runs"]
    for r in runs:
        if r.get("meta", {}).get("dataset") == dataset:
            return r["run_id"]
    return None


def test_library_loaded(page) -> None:
    print("=== A. Library populated, Acquisition Control replaced by Replay Control ===")
    page.goto(BASE, wait_until="load")
    page.wait_for_function("() => document.querySelectorAll('#runSelect option').length > 1", timeout=15000)
    # The old live dataset/bearing selects are hidden; the run picker drives it.
    assert page.is_hidden("#dataset"), "dataset select should be hidden"
    assert page.text_content(".panel-ctl .ttl").strip() == "Replay Control"
    # Run Library + Multi-tier PdM panels were removed.
    assert page.query_selector("#btnRecordAll") is None, "Run Library should be gone"
    assert page.query_selector("#tierMini") is None, "Multi-tier PdM should be gone"
    # Dead IG stream controls removed; panel is drop-anchored.
    assert page.query_selector("#btnAutoIg") is None, "dead Auto IG control should be gone"
    assert page.query_selector("#btnExplain") is None, "dead Refresh-now control should be gone"
    opts = page.eval_on_selector_all("#runSelect option", "els => els.length")
    print(f"  picker options={opts} (Run Library + Multi-tier PdM removed)")
    assert opts > 1, "run picker not populated"
    print("  OK")


def test_pick_plays_from_start(page) -> None:
    print("=== B. Picking a run plays it from acquisition 0 with drops aligned ===")
    run_id = _pick_run("xjtusy") or _pick_run("phm2012")
    assert run_id, "no run to pick"
    page.select_option("#runSelect", run_id)
    page.wait_for_function(
        "() => /Replaying/.test(document.getElementById('statusText').textContent)", timeout=10000
    )
    started = _acq(page)
    drops = len(page.query_selector_all(".drop-row"))
    print(f"  playing {run_id}: started at acq {started}, drops listed {drops}")
    assert started <= 5, "should play from the start"
    assert not page.is_hidden("#failZone"), "failure-analysis zone should show"
    time.sleep(1.0)
    assert _acq(page) > started, "playback should advance"
    print("  OK")


def test_player_controls_and_scrub(page) -> None:
    print("=== C. Pause / Seek / Step / Reset ===")
    page.click("#btnPause")
    page.wait_for_function(
        "() => /Replay paused/.test(document.getElementById('statusText').textContent)", timeout=5000
    )
    e0 = page.text_content("#mElapsed")
    set_slider(page, 8)
    time.sleep(0.2)
    seen = _acq(page)
    e1 = page.text_content("#mElapsed")
    print(f"  scrubbed to 8 → acq {seen}, elapsed {e0!r} → {e1!r}")
    assert seen in (8, 9) and e1 != e0, "seek did not read values at that acquisition"
    page.click("#btnStep")
    time.sleep(0.1)
    assert _acq(page) == seen + 1, "Step should advance one acquisition"
    page.click("#btnReset")
    time.sleep(0.1)
    assert _acq(page) <= 1, "Reset should return to start"
    print("  OK")


def test_drop_explanation(page) -> None:
    print("=== D. Drop explanation ===")
    rows = page.query_selector_all(".drop-row")
    if not rows:
        print("  (no drops in this run — skipping)")
        return
    cap_before = page.text_content("#igCaption")
    assert "No drop" in (cap_before or ""), f"IG should be empty before a drop is picked, got {cap_before!r}"
    rows[0].click()
    page.wait_for_function(
        "() => /IG @ drop/.test(document.getElementById('igCaption').textContent)", timeout=10000
    )
    print(f"    IG caption: {page.text_content('#igCaption')!r}")
    page.wait_for_function(
        "() => document.querySelectorAll('#hiDelta .hid-row').length > 0", timeout=10000
    )
    headline = page.text_content("#dropHeadline")
    print(f"    headline: {headline!r}")
    assert "Health fell" in (headline or "")

    # Link to Health Indicator Trend: the top changed feature auto-overlays
    # (one row is active), and clicking another row moves the overlay.
    active0 = page.eval_on_selector_all("#hiDelta .hid-row.is-active", "els => els.length")
    hid_rows = page.query_selector_all("#hiDelta .hid-row")
    assert active0 == 1, f"expected the top HI feature auto-plotted on the HI chart, got {active0} active"
    if len(hid_rows) > 1:
        hid_rows[1].click()
        page.wait_for_timeout(150)
        active_name = page.eval_on_selector("#hiDelta .hid-row.is-active", "el => el.dataset.name")
        clicked_name = hid_rows[1].get_attribute("data-name")
        print(f"    clicked feature now plotted on HI chart: {active_name}")
        assert active_name == clicked_name, "clicking a feature did not move the HI-chart overlay"
    print("  OK")


def test_play_restart(page) -> None:
    print("=== E. Play restarts the loaded run from 0 ===")
    page.click("#btnStart")  # ▶ Play
    page.wait_for_function(
        "() => /Replaying/.test(document.getElementById('statusText').textContent)", timeout=5000
    )
    time.sleep(0.2)
    assert _acq(page) <= 6, "Play should restart from the beginning"
    page.click("#btnPause")
    print("  OK")


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 950})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else None)
        try:
            test_library_loaded(page)
            test_pick_plays_from_start(page)
            test_player_controls_and_scrub(page)
            test_drop_explanation(page)
            test_play_restart(page)
        finally:
            browser.close()
        if errors:
            print("\nJS PAGE ERRORS:")
            for e in errors:
                print("  ", e)
            return 1
    print("\nAll browser tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
