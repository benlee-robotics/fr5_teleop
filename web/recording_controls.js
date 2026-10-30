// Quest A/B buttons → demonstration recording commands, plus the in-headset
// status light. Pure logic (no DOM/WebXR access) so it can be tested headless.
//
//   A short          IDLE: start (server uses the next task set from /monitor)
//                    RECORDING: stop and save as success
//   B short          RECORDING: stop and save as failure (fires on release)
//   B hold >= 1 s    RECORDING: discard the episode
//
// Any other combination returns {action: 'rejected'} for error feedback.
// Recording never moves the robot; grip/trigger handling is unchanged.

const RECORD_LONG_PRESS_MS = 1000;
// Ignore further commands briefly so a double press cannot start and
// immediately stop an episode before the next status arrives.
const RECORD_COOLDOWN_MS = 600;

class RecordButtons {
  constructor() {
    this.reset();
  }

  reset() {
    this.aWasPressed = false;
    this.bWasPressed = false;
    this.bDownMs = null;
    this.bLongFired = false;
    this.cooldownUntilMs = 0;
  }

  // recorderState: 'IDLE' | 'RECORDING' | 'FINALIZING' | 'DOWN' | null
  // (null when recording is disabled). Returns null or a command object.
  update(aPressed, bPressed, nowMs, recorderState) {
    let command = null;

    if (aPressed && !this.aWasPressed) {
      if (recorderState === 'IDLE') {
        command = { action: 'start' };
      } else if (recorderState === 'RECORDING') {
        command = { action: 'stop', outcome: 'success' };
      } else {
        command = { action: 'rejected', reason: recorderState || 'disabled' };
      }
    }

    if (bPressed && !this.bWasPressed) {
      this.bDownMs = nowMs;
      this.bLongFired = false;
    }
    if (
      command === null
      && bPressed
      && this.bDownMs !== null
      && !this.bLongFired
      && nowMs - this.bDownMs >= RECORD_LONG_PRESS_MS
    ) {
      this.bLongFired = true;
      command = recorderState === 'RECORDING'
        ? { action: 'discard' }
        : { action: 'rejected', reason: recorderState || 'disabled' };
    }
    if (!bPressed && this.bWasPressed) {
      if (command === null && this.bDownMs !== null && !this.bLongFired) {
        command = recorderState === 'RECORDING'
          ? { action: 'stop', outcome: 'failure' }
          : { action: 'rejected', reason: recorderState || 'disabled' };
      }
      this.bDownMs = null;
    }

    this.aWasPressed = aPressed;
    this.bWasPressed = bPressed;
    if (command !== null && command.action !== 'rejected') {
      if (nowMs < this.cooldownUntilMs) return null;
      this.cooldownUntilMs = nowMs + RECORD_COOLDOWN_MS;
    }
    return command;
  }
}

// Haptic patterns: [intensity 0..1, duration ms, pause ms after].
const RECORD_HAPTICS = {
  start: [[0.6, 80, 0]],
  success: [[0.6, 80, 120], [0.6, 80, 0]],
  failure: [[0.8, 300, 0]],
  discard: [[0.5, 60, 100], [0.5, 60, 100], [0.5, 60, 0]],
  error: [[1.0, 500, 0]],
};

const FLASH_MS = 1500;
const LIGHT_COLORS = {
  recording: [0.95, 0.1, 0.1],
  finalizing: [0.95, 0.8, 0.1],
  success: [0.1, 0.85, 0.2],
  failure: [1.0, 0.5, 0.0],
  interrupted: [0.9, 0.9, 0.9],
  discard: [0.6, 0.6, 0.6],
  error: [0.9, 0.1, 0.9],
};

// Color [r, g, b] of the head-locked status light, or null for none.
// flash: {color: key of LIGHT_COLORS, untilMs} shown on top of the state.
function recordLightColor(recorderState, flash, nowMs) {
  if (flash && nowMs < flash.untilMs) {
    // Blink errors so they are not mistaken for a recording light.
    if (flash.color === 'error' && nowMs % 300 >= 150) return null;
    return LIGHT_COLORS[flash.color] || null;
  }
  if (recorderState === 'RECORDING') {
    return nowMs % 1000 < 700 ? LIGHT_COLORS.recording : null;
  }
  if (recorderState === 'FINALIZING') return LIGHT_COLORS.finalizing;
  return null;
}

// Watches recorder status to produce flashes for saves and refused commands.
class RecordStatusTracker {
  constructor() {
    this.lastSavedIndex = undefined;
    this.lastErrorSeq = undefined;
    this.pendingUntilMs = 0;
  }

  commandSent(nowMs) {
    // Errors raised within this window are attributed to our command.
    this.pendingUntilMs = nowMs + 2000;
  }

  // Returns {flash, haptic} or null.
  observe(recording, nowMs) {
    if (!recording) return null;
    const saved = recording.last_saved || null;
    const savedIndex = saved ? saved.episode_index : null;
    // error_seq increments on every recorder error, even a repeated text.
    const errorSeq = Number.isInteger(recording.error_seq) ? recording.error_seq : 0;
    let result = null;
    if (this.lastSavedIndex !== undefined && savedIndex !== this.lastSavedIndex && saved) {
      const color = saved.outcome in LIGHT_COLORS ? saved.outcome : 'success';
      result = { flash: { color, untilMs: nowMs + FLASH_MS }, haptic: null };
    }
    if (
      this.lastErrorSeq !== undefined
      && errorSeq > this.lastErrorSeq
      && nowMs < this.pendingUntilMs
    ) {
      result = { flash: { color: 'error', untilMs: nowMs + FLASH_MS }, haptic: 'error' };
    }
    this.lastSavedIndex = savedIndex;
    this.lastErrorSeq = errorSeq;
    return result;
  }
}
