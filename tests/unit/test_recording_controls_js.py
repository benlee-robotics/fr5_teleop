"""Runs web/recording_controls.js in headless Chrome (skipped without Chrome)."""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


WEB = Path(__file__).resolve().parents[2] / "web"

JS_TESTS = r"""
const results = [];
function check(name, actual, expected) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  results.push({ name, ok: a === e, actual: a, expected: e });
}

// Drive a RecordButtons instance through (a, b, t, state) steps and collect
// every non-null command.
function run(steps) {
  const buttons = new RecordButtons();
  const out = [];
  for (const [a, b, t, state] of steps) {
    const cmd = buttons.update(a, b, t, state);
    if (cmd) out.push(Object.assign({ t }, cmd));
  }
  return out;
}

check('A in IDLE starts', run([[true, false, 0, 'IDLE']]),
  [{ t: 0, action: 'start' }]);
check('A held does not repeat', run([
  [true, false, 0, 'IDLE'], [true, false, 16, 'RECORDING'], [true, false, 900, 'RECORDING'],
]), [{ t: 0, action: 'start' }]);
check('A in RECORDING saves success', run([[true, false, 0, 'RECORDING']]),
  [{ t: 0, action: 'stop', outcome: 'success' }]);
check('double A within cooldown is swallowed', run([
  [true, false, 0, 'IDLE'], [false, false, 100, 'RECORDING'], [true, false, 200, 'RECORDING'],
]), [{ t: 0, action: 'start' }]);
check('second A after cooldown stops', run([
  [true, false, 0, 'IDLE'], [false, false, 100, 'RECORDING'], [true, false, 700, 'RECORDING'],
]), [{ t: 0, action: 'start' }, { t: 700, action: 'stop', outcome: 'success' }]);
check('B short saves failure on release', run([
  [false, true, 0, 'RECORDING'], [false, true, 400, 'RECORDING'], [false, false, 500, 'RECORDING'],
]), [{ t: 500, action: 'stop', outcome: 'failure' }]);
check('B hold discards once, release does nothing', run([
  [false, true, 0, 'RECORDING'], [false, true, 999, 'RECORDING'],
  [false, true, 1000, 'RECORDING'], [false, true, 1500, 'RECORDING'],
  [false, false, 1600, 'IDLE'],
]), [{ t: 1000, action: 'discard' }]);
check('A while not ready is rejected', run([[true, false, 0, 'FINALIZING']]),
  [{ t: 0, action: 'rejected', reason: 'FINALIZING' }]);
check('A with recording disabled is rejected', run([[true, false, 0, null]]),
  [{ t: 0, action: 'rejected', reason: 'disabled' }]);
check('B in IDLE is rejected on release', run([
  [false, true, 0, 'IDLE'], [false, false, 200, 'IDLE'],
]), [{ t: 200, action: 'rejected', reason: 'IDLE' }]);
check('B hold in IDLE rejects once', run([
  [false, true, 0, 'IDLE'], [false, true, 1000, 'IDLE'], [false, false, 1200, 'IDLE'],
]), [{ t: 1000, action: 'rejected', reason: 'IDLE' }]);
check('rejections are not rate limited', run([
  [true, false, 0, 'DOWN'], [false, false, 50, 'DOWN'], [true, false, 100, 'DOWN'],
]), [{ t: 0, action: 'rejected', reason: 'DOWN' }, { t: 100, action: 'rejected', reason: 'DOWN' }]);

check('light blinks red while recording',
  [recordLightColor('RECORDING', null, 100), recordLightColor('RECORDING', null, 800)],
  [LIGHT_COLORS.recording, null]);
check('light off when idle', recordLightColor('IDLE', null, 0), null);
check('flash overrides state', recordLightColor('RECORDING', { color: 'success', untilMs: 50 }, 10),
  LIGHT_COLORS.success);
check('expired flash falls back to state', recordLightColor('FINALIZING', { color: 'success', untilMs: 5 }, 10),
  LIGHT_COLORS.finalizing);

const tracker = new RecordStatusTracker();
check('first status only primes', tracker.observe({ error_seq: 3, last_saved: null }, 0), null);
check('stale error without command is ignored', tracker.observe({ error_seq: 4, last_saved: null }, 10), null);
tracker.commandSent(100);
const refused = tracker.observe({ error_seq: 5, last_saved: null }, 200);
check('new error after command flashes', [refused.flash.color, refused.haptic], ['error', 'error']);
tracker.commandSent(300);
const repeated = tracker.observe({ error_seq: 6, last_saved: null }, 400);
check('repeated error text still detected', repeated && repeated.haptic, 'error');
const saved = tracker.observe({ error_seq: 6, last_saved: { episode_index: 2, outcome: 'failure' } }, 500);
check('save flashes outcome color', saved.flash.color, 'failure');
check('same save does not flash twice',
  tracker.observe({ error_seq: 6, last_saved: { episode_index: 2, outcome: 'failure' } }, 600), null);

document.getElementById('out').textContent = JSON.stringify(results);
"""


def find_chrome() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return None


def test_recording_controls_in_headless_chrome(tmp_path: Path) -> None:
    chrome = find_chrome()
    if chrome is None:
        pytest.skip("Chrome is not installed")
    page = tmp_path / "test.html"
    page.write_text(
        "<!doctype html><meta charset=utf-8><pre id=out>NOT RUN</pre>"
        f"<script src='{(WEB / 'recording_controls.js').as_uri()}'></script>"
        f"<script>{JS_TESTS}</script>",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--allow-file-access-from-files",
            f"--user-data-dir={tmp_path / 'profile'}",
            "--dump-dom",
            page.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    match = re.search(r'<pre id="out">(.*?)</pre>', result.stdout, re.S)
    assert match, f"no output from Chrome: {result.stdout[-500:]} {result.stderr[-500:]}"
    output = html.unescape(match.group(1))
    assert output != "NOT RUN", "JS tests threw before finishing"
    results = json.loads(output)
    failures = [r for r in results if not r["ok"]]
    assert not failures, failures
    assert len(results) == 22
