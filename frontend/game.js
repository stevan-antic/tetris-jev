"use strict";

/*
 * Tetris web client — human mode plus the Jev (LLM) player-mode toggle.
 *
 * All game logic lives on the server. This client:
 *   - connects over WebSocket and renders server-authoritative snapshots,
 *   - translates keyboard input into commands (with client-side DAS/ARR),
 *   - acks every snapshot, detects seq gaps and sends resync,
 *   - reconnects with 1–2s backoff and resyncs on reconnect,
 *   - switches between "human" and "jev" player modes via {"type":"set_mode"},
 *     with the server snapshot's "mode" key authoritative on the active mode,
 *   - adjusts the Jev decision pause via {"type":"set_jev_pause","seconds":N},
 *     with the server snapshot's "jev_pause" key authoritative on the value,
 *   - picks the LLM brain profile via {"type":"set_brain","profile":"<id>"},
 *     with the server snapshot's "brain"/"profiles" keys authoritative on the
 *     active profile and the selectable list,
 *   - sets the start level via {"type":"set_level","level":N} (1..10), which
 *     makes the server restart the game at that level.
 *
 * Protocol (see PLAN.md):
 *   client → server: {"type":"command","cmd":"move_left"|...,"ts":ms}
 *                    (soft drop uses {"type":"command","cmd":"soft_drop","phase":"down"|"up","ts":ms})
 *                    {"type":"ack","seq":N}      after each accepted snapshot
 *                    {"type":"resync"}           on seq gap / after reconnect
 *                    {"type":"set_mode","mode":"human"|"jev"}
 *                    {"type":"set_jev_pause","seconds":N}   (clamped to [0,30])
 *                    {"type":"set_brain","profile":"<id>"}
 *                    {"type":"set_level","level":N}   (start level 1..10; server resets the game)
 *   server → client: {"type":"snapshot","seq":N,"board":[20 rows],"hold":char|null,
 *                     "next":[5 chars],"score":N,"lines":N,"level":N,"combo":N,
 *                     "b2b":bool,"game_over":bool,"jev_pause":N,
 *                     "brain":"<id>","profiles":[{id,available},...]}
 *                    {"type":"game_over"}, {"type":"ack"}, {"type":"resync"}
 *
 * The falling piece is NOT part of the snapshot contract. If the server
 * includes an optional "piece" field ({type,x,y,rot} — x/y top-left of the
 * piece's bounding box), it is rendered on top of the locked cells; if the
 * field is absent (or unparseable) the client renders locked cells only.
 *
 * Player mode is likewise optional on the wire for backwards compatibility:
 * a snapshot without "mode" is treated as "human". In "jev" mode the server
 * may add a "ghost" field ({type,x,y,rot} of the piece's target landing
 * placement, same coordinates as "piece" — the client draws it at ~35% alpha
 * beneath the falling piece) and the client suppresses all keyboard commands.
 *
 * The Jev decision pause ("jev_pause" — seconds the server waits between Jev
 * decisions) is carried in every snapshot that includes it; a snapshot
 * without the key leaves the current value unchanged. The client changes it
 * with {"type":"set_jev_pause","seconds":N} (clamped to [0,30] on both sides)
 * and, like the player mode, shows the requested value dimmed until a server
 * snapshot confirms it.
 *
 * The brain profile ("brain" — the LLM backend the Jev player uses) and the
 * selectable profile list ("profiles" — [{id,available},...] straight from the
 * server, so env.yaml changes flow through) work the same way: the client
 * picks a profile with {"type":"set_brain","profile":"<id>"}. Unavailable or
 * unknown profiles get no ack from the server; the request shows dimmed and
 * the next snapshot clears the pending state, reverting the dropdown to the
 * snapshot's "brain". Unavailable profiles render as disabled options.
 *
 * The start level ("level" in the snapshot) is set with
 * {"type":"set_level","level":N} (clamped to 1..10 client-side; the server
 * clamps too): the server RESETS the game with that start level, so the next
 * snapshot carries the fresh board with level=N. The request shows dimmed
 * until that snapshot arrives, same reconciliation as the pause control.
 */

(() => {
  // ------------------------------------------------------------------
  // Constants
  // ------------------------------------------------------------------

  const COLS = 10;
  const ROWS = 20;
  const CELL = 32; // board canvas cell size in px (canvas is 320x640)
  const FRAME_MS = 1000 / 60;

  const DAS_MS = 167; // delayed auto shift
  const ARR_MS = 33;  // auto repeat rate

  const WS_PATH = "/ws";
  const RECONNECT_MIN_MS = 1000;
  const RECONNECT_MAX_MS = 2000;

  const PAUSE_STEP = 0.5; // seconds per +/- click on the pause control
  const PAUSE_MIN = 0;    // client-side clamp; the server clamps to [0, 30] too
  const PAUSE_MAX = 30;

  const LEVEL_MIN = 1;    // start-level stepper bounds (client-side clamp;
  const LEVEL_MAX = 10;   // the server clamps to [1, 10] too)

  const COLORS = {
    I: "#22d3ee",
    J: "#3b82f6",
    L: "#f59e0b",
    O: "#eab308",
    S: "#22c55e",
    T: "#a855f7",
    Z: "#ef4444",
  };

  // Spawn orientation, as (row, col) cells inside each piece's SRS box.
  const SPAWN_CELLS = {
    I: [[1, 0], [1, 1], [1, 2], [1, 3]],
    J: [[0, 0], [1, 0], [1, 1], [1, 2]],
    L: [[0, 2], [1, 0], [1, 1], [1, 2]],
    O: [[0, 0], [0, 1], [1, 0], [1, 1]],
    S: [[0, 1], [0, 2], [1, 0], [1, 1]],
    T: [[0, 1], [1, 0], [1, 1], [1, 2]],
    Z: [[0, 0], [0, 1], [1, 1], [1, 2]],
  };
  const BOX = { I: 4, J: 3, L: 3, O: 2, S: 3, T: 3, Z: 3 };

  // Precompute all 4 rotation states per piece (clockwise within the box).
  const ROTATIONS = (() => {
    const out = {};
    for (const type of Object.keys(SPAWN_CELLS)) {
      let cells = SPAWN_CELLS[type];
      const states = [cells];
      const n = BOX[type];
      for (let i = 0; i < 3; i += 1) {
        cells = cells.map(([r, c]) => [c, n - 1 - r]);
        states.push(cells);
      }
      out[type] = states;
    }
    return out;
  })();

  const GAME_KEYS = new Set([
    "ArrowLeft", "ArrowRight", "ArrowDown", "ArrowUp",
    "KeyX", "KeyZ", "KeyC", "ShiftLeft", "ShiftRight",
    "Space", "KeyP", "KeyR",
  ]);

  // ------------------------------------------------------------------
  // DOM
  // ------------------------------------------------------------------

  const boardCanvas = document.getElementById("board");
  const bctx = boardCanvas.getContext("2d");
  const holdCanvas = document.getElementById("hold");
  const hctx = holdCanvas.getContext("2d");
  const nextCanvas = document.getElementById("next");
  const nctx = nextCanvas.getContext("2d");

  const el = {
    score: document.getElementById("score"),
    lines: document.getElementById("lines"),
    level: document.getElementById("level"),
    combo: document.getElementById("combo"),
    b2b: document.getElementById("b2b"),
    conn: document.getElementById("conn"),
    connText: document.getElementById("conn-text"),
    modeBadge: document.getElementById("mode-badge"),
    modeToggle: document.getElementById("mode-toggle"),
    modeBtnHuman: document.getElementById("mode-btn-human"),
    modeBtnJev: document.getElementById("mode-btn-jev"),
    pauseControl: document.getElementById("pause-control"),
    pauseValue: document.getElementById("pause-value"),
    pauseDec: document.getElementById("pause-dec"),
    pauseInc: document.getElementById("pause-inc"),
    brainControl: document.getElementById("brain-control"),
    brainSelect: document.getElementById("brain-select"),
    levelControl: document.getElementById("level-control"),
    levelValue: document.getElementById("level-value"),
    levelDec: document.getElementById("level-dec"),
    levelInc: document.getElementById("level-inc"),
    overlayPause: document.getElementById("overlay-pause"),
    overlayOver: document.getElementById("overlay-over"),
    finalScore: document.getElementById("final-score"),
    finalLines: document.getElementById("final-lines"),
  };

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------

  const state = {
    board: null,      // 20 strings of '.'/'IJLOSTZ' (locked cells, row 0 = top)
    piece: null,      // optional server-provided falling piece
    ghost: null,      // jev mode: target landing placement {type,x,y,rot} or null
    mode: "human",    // active player mode, authoritative from the server
    jevPause: 1.0,    // seconds between Jev decisions, authoritative from snapshots
    brain: null,      // active brain profile id, authoritative from snapshots
    profiles: [],     // selectable profiles [{id, available}], server list
    hold: null,       // held piece char or null
    next: [],         // up to 5 piece chars
    score: 0,
    lines: 0,
    level: 1,
    combo: 0,
    b2b: false,
    over: false,
    dirty: true,      // side panels need re-render
  };

  let paused = false;
  let pausedAt = 0;
  // Mode the toggle has requested but the server has not confirmed yet.
  // null means there is no in-flight switch. The server snapshot is
  // authoritative: state.mode only changes when a snapshot's "mode" matches.
  let pendingMode = null;

  // Pause seconds the +/- control has requested but no snapshot has confirmed
  // yet. null means no change is in flight. The server snapshot is
  // authoritative: state.jevPause only changes when a snapshot carries it.
  let pendingPause = null;

  // Brain profile the dropdown has requested but no snapshot has confirmed
  // yet. null means no switch is in flight. The server snapshot is
  // authoritative: state.brain only changes when a snapshot carries "brain".
  // Unavailable/unknown profiles get NO ack, so the next snapshot clears the
  // pending state and the dropdown reverts to the snapshot's "brain" — the
  // same reconciliation as the player-mode toggle.
  let pendingBrain = null;

  // Signature of the profiles list the dropdown options were built from, so
  // the option list is only rebuilt when the server's list actually changes.
  let brainOptionsSig = "";

  // Start level the +/- stepper has requested but no snapshot has confirmed
  // yet. null means no reset is in flight. set_level makes the server restart
  // the game at that level; the next snapshot carries the fresh board with
  // level=N, which is the reconciliation.
  let pendingLevel = null;

  // Client-side DAS/ARR for left/right movement.
  const das = {
    dir: null,      // currently active direction ("left" | "right" | null)
    leftHeld: false,
    rightHeld: false,
    start: 0,       // timestamp when the current charge began
    charged: false, // DAS delay elapsed, ARR ticking
    next: 0,        // timestamp of the next ARR move
  };
  let softHeld = false;

  // ------------------------------------------------------------------
  // Connection
  // ------------------------------------------------------------------

  const conn = {
    ws: null,
    retryDelay: RECONNECT_MIN_MS,
    timer: null,
  };
  let lastSeq = null;

  function setStatus(kind) {
    el.conn.className = "conn " + kind;
    el.connText.textContent =
      kind === "connected" ? "Connected"
      : kind === "connecting" ? "Connecting…"
      : kind === "reconnecting" ? "Reconnecting…"
      : "Disconnected";
  }

  function connect() {
    setStatus("connecting");
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    let ws;
    try {
      ws = new WebSocket(proto + "//" + location.host + WS_PATH);
    } catch (err) {
      scheduleReconnect();
      return;
    }
    conn.ws = ws;

    ws.onopen = () => {
      conn.retryDelay = RECONNECT_MIN_MS;
      setStatus("connected");
      lastSeq = null;
      // Server replies to resync with a full snapshot.
      send({ type: "resync" });
    };

    ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch (err) {
        return;
      }
      handleMessage(msg);
    };

    ws.onerror = () => { /* onclose follows */ };

    ws.onclose = () => {
      conn.ws = null;
      setStatus("reconnecting");
      scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    if (conn.timer) return;
    const delay = conn.retryDelay;
    conn.retryDelay = Math.min(RECONNECT_MAX_MS, conn.retryDelay * 1.5);
    conn.timer = setTimeout(() => {
      conn.timer = null;
      connect();
    }, delay);
  }

  function send(obj) {
    if (conn.ws && conn.ws.readyState === WebSocket.OPEN) {
      conn.ws.send(JSON.stringify(obj));
    }
  }

  function sendCommand(cmd, extra) {
    send(Object.assign({ type: "command", cmd, ts: Date.now() }, extra || {}));
  }

  // ------------------------------------------------------------------
  // Player mode (human / jev)
  // ------------------------------------------------------------------

  /*
   * Reflects the mode state onto the toggle buttons and the badge. While a
   * switch is pending (requested but not yet confirmed by a snapshot) the
   * requested mode is shown dimmed; the server snapshot is authoritative.
   */
  function updateModeUI() {
    const active = pendingMode !== null ? pendingMode : state.mode;
    const isPending = pendingMode !== null;
    const isJev = active === "jev";

    el.modeBadge.textContent = isJev ? "Jev" : "Human";
    el.modeBadge.classList.toggle("jev", isJev);
    el.modeBadge.classList.toggle("human", !isJev);
    el.modeBadge.classList.toggle("pending", isPending);

    el.modeBtnHuman.classList.toggle("active", !isJev);
    el.modeBtnJev.classList.toggle("active", isJev);
    el.modeToggle.classList.toggle("pending", isPending);
  }

  function requestMode(mode) {
    if (mode !== "human" && mode !== "jev") return;
    if (mode === state.mode && pendingMode === null) return; // already active
    if (mode === pendingMode) return; // already requesting this switch
    pendingMode = mode;
    send({ type: "set_mode", mode });
    updateModeUI();
  }

  el.modeBtnHuman.addEventListener("click", () => requestMode("human"));
  el.modeBtnJev.addEventListener("click", () => requestMode("jev"));

  // ------------------------------------------------------------------
  // Jev decision pause (seconds the server waits between Jev decisions)
  // ------------------------------------------------------------------

  /*
   * Value currently shown: the pending request while one is in flight, else
   * the authoritative snapshot value. Like the mode toggle, the pending value
   * is shown dimmed until the server snapshot confirms it.
   */
  function displayedPause() {
    return pendingPause !== null ? pendingPause : state.jevPause;
  }

  function clampPause(seconds) {
    return Math.min(PAUSE_MAX, Math.max(PAUSE_MIN, seconds));
  }

  /*
   * Reflects the pause state onto the control: value readout, pending dim and
   * the +/- button bounds. In human mode the control is styled dimmer (it
   * only affects Jev) but stays clickable so the value can be pre-set — the
   * server stores it regardless of the active mode.
   */
  function updatePauseUI() {
    const value = displayedPause();
    el.pauseValue.textContent = value.toFixed(1) + "s";
    el.pauseControl.classList.toggle("pending", pendingPause !== null);
    el.pauseControl.classList.toggle("human-mode", state.mode !== "jev");
    el.pauseDec.disabled = value <= PAUSE_MIN;
    el.pauseInc.disabled = value >= PAUSE_MAX;
  }

  function requestPause(seconds) {
    const target = clampPause(seconds);
    if (target === displayedPause()) return; // already the active/requested value
    pendingPause = target;
    send({ type: "set_jev_pause", seconds: target });
    updatePauseUI();
  }

  el.pauseDec.addEventListener("click", () => {
    requestPause(displayedPause() - PAUSE_STEP);
  });
  el.pauseInc.addEventListener("click", () => {
    requestPause(displayedPause() + PAUSE_STEP);
  });

  // ------------------------------------------------------------------
  // Brain profile (the LLM backend the Jev player uses)
  // ------------------------------------------------------------------

  /*
   * Rebuilds the dropdown options from the server's "profiles" list whenever
   * that list changes — there is no hardcoded profile list, so env.yaml
   * changes flow through. Unavailable profiles render as disabled options
   * ("<id> (unavailable)").
   */
  function rebuildBrainOptions() {
    const sig = state.profiles
      .map((p) => p.id + (p.available ? "" : "!"))
      .join("|");
    if (sig === brainOptionsSig) return; // list unchanged — keep the options
    brainOptionsSig = sig;
    el.brainSelect.textContent = "";
    for (const p of state.profiles) {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.available ? p.id : p.id + " (unavailable)";
      opt.disabled = !p.available;
      el.brainSelect.appendChild(opt);
    }
  }

  /*
   * Profile currently shown: the pending request while one is in flight,
   * else the authoritative snapshot value. Like the mode toggle, a pending
   * switch is dimmed until the next snapshot confirms (or reverts) it.
   */
  function displayedBrain() {
    return pendingBrain !== null ? pendingBrain : state.brain;
  }

  function updateBrainUI() {
    rebuildBrainOptions();
    const shown = displayedBrain();
    if (shown !== null && el.brainSelect.value !== shown) {
      el.brainSelect.value = shown; // no-op for unknown ids (no matching option)
    }
    el.brainControl.classList.toggle("pending", pendingBrain !== null);
  }

  el.brainSelect.addEventListener("change", () => {
    const id = el.brainSelect.value;
    if (typeof id !== "string" || id === "" || id === displayedBrain()) return;
    pendingBrain = id;
    send({ type: "set_brain", profile: id });
    el.brainSelect.blur(); // arrows/space stay wired to the game, not the select
    updateBrainUI();
  });

  // ------------------------------------------------------------------
  // Start level stepper (set_level restarts the game at that level)
  // ------------------------------------------------------------------

  function displayedLevel() {
    return pendingLevel !== null ? pendingLevel : state.level;
  }

  function clampLevel(level) {
    return Math.min(LEVEL_MAX, Math.max(LEVEL_MIN, level));
  }

  /*
   * Reflects the start level onto the stepper: value readout, pending dim
   * and the +/- button bounds. The request stays dimmed until the snapshot
   * with the fresh board (level=N) arrives.
   */
  function updateLevelUI() {
    el.levelValue.textContent = String(displayedLevel());
    el.levelControl.classList.toggle("pending", pendingLevel !== null);
    el.levelDec.disabled = displayedLevel() <= LEVEL_MIN;
    el.levelInc.disabled = displayedLevel() >= LEVEL_MAX;
  }

  function requestLevel(level) {
    const target = clampLevel(Math.round(level));
    if (!Number.isFinite(target)) return;
    if (target === displayedLevel()) return; // already the active/requested value
    pendingLevel = target;
    send({ type: "set_level", level: target });
    updateLevelUI();
  }

  el.levelDec.addEventListener("click", () => {
    el.levelDec.blur(); // keep arrows/space wired to the game, not the stepper
    requestLevel(displayedLevel() - 1);
  });
  el.levelInc.addEventListener("click", () => {
    el.levelInc.blur();
    requestLevel(displayedLevel() + 1);
  });

  // ------------------------------------------------------------------
  // Message handling
  // ------------------------------------------------------------------

  function handleMessage(msg) {
    if (!msg || typeof msg !== "object") return;
    switch (msg.type) {
      case "snapshot":
        onSnapshot(msg);
        break;
      case "game_over":
        onGameOver(msg);
        break;
      case "ack":      // server confirmations — nothing to do
      case "resync":
      default:
        break;
    }
  }

  function onSnapshot(snap) {
    const seq = typeof snap.seq === "number" ? snap.seq : null;

    if (seq !== null) {
      if (lastSeq !== null && seq > lastSeq + 1) {
        // Gap detected: drop this partial snapshot, reset tracking and
        // ask the server for a full snapshot.
        lastSeq = null;
        send({ type: "resync" });
        return;
      }
      if (lastSeq !== null && seq <= lastSeq) {
        return; // stale or duplicate snapshot
      }
      lastSeq = seq;
      send({ type: "ack", seq });
    }

    state.board = Array.isArray(snap.board) ? snap.board : null;
    state.piece = snap.piece !== undefined ? snap.piece : null;
    state.ghost = snap.ghost !== undefined ? snap.ghost : null;
    state.hold = typeof snap.hold === "string" ? snap.hold : null;
    state.next = Array.isArray(snap.next) ? snap.next : [];
    state.score = typeof snap.score === "number" ? snap.score : state.score;
    state.lines = typeof snap.lines === "number" ? snap.lines : state.lines;
    state.level = typeof snap.level === "number" ? snap.level : state.level;
    state.combo = typeof snap.combo === "number" ? snap.combo : 0;
    state.b2b = snap.b2b === true;
    state.dirty = true;

    // Server is authoritative on the active player mode; a missing "mode"
    // key is treated as "human" (pre-Jev servers simply omit it).
    const snapMode = typeof snap.mode === "string" &&
      (snap.mode === "human" || snap.mode === "jev") ? snap.mode : "human";
    if (state.mode !== snapMode) {
      state.mode = snapMode;
      if (snapMode === "jev") clearHeldState(); // keep no keys stuck in jev mode
    }
    pendingMode = null; // snapshot reconciles any pending switch
    updateModeUI();

    // Server is authoritative on the Jev decision pause; a missing
    // "jev_pause" key keeps the current value (pre-pause servers omit it).
    if (typeof snap.jev_pause === "number" && Number.isFinite(snap.jev_pause)) {
      state.jevPause = snap.jev_pause;
    }
    pendingPause = null; // snapshot reconciles any pending pause change
    updatePauseUI();

    // Server is authoritative on the brain profile and the selectable list;
    // a missing "brain" or "profiles" key keeps the current values (pre-brain
    // servers omit them). Unavailable/unknown profiles get no ack from the
    // server — the snapshot never changes its "brain", so clearing
    // pendingBrain here reverts the dropdown to the authoritative value.
    if (typeof snap.brain === "string" && snap.brain !== "") {
      state.brain = snap.brain;
    }
    if (Array.isArray(snap.profiles)) {
      state.profiles = snap.profiles.filter((p) =>
        p !== null && typeof p === "object" && typeof p.id === "string");
    }
    pendingBrain = null; // snapshot reconciles any pending switch
    updateBrainUI();

    // The fresh board after a set_level reset carries the new start level.
    pendingLevel = null; // snapshot reconciles any pending reset
    updateLevelUI();

    if (snap.game_over === true) {
      onGameOver(snap);
    }
  }

  function onGameOver(msg) {
    if (state.over) return;
    state.over = true;
    clearHeldState();
    if (typeof msg.score === "number") state.score = msg.score;
    if (typeof msg.lines === "number") state.lines = msg.lines;
    el.finalScore.textContent = String(state.score);
    el.finalLines.textContent = String(state.lines);
    el.overlayOver.classList.remove("hidden");
  }

  // ------------------------------------------------------------------
  // Input
  // ------------------------------------------------------------------

  function now() {
    return performance.now();
  }

  function pressDir(dir) {
    das[dir + "Held"] = true;
    das.dir = dir; // last-pressed wins
    das.start = now();
    das.charged = false;
    sendCommand(dir === "left" ? "move_left" : "move_right");
  }

  function releaseDir(dir) {
    das[dir + "Held"] = false;
    if (das.dir !== dir) return;
    const other = dir === "left" ? "right" : "left";
    if (das[other + "Held"]) {
      // Return to the still-held direction: re-charge DAS from zero,
      // no immediate move (that would double-move the piece).
      das.dir = other;
      das.start = now();
      das.charged = false;
    } else {
      das.dir = null;
      das.charged = false;
    }
  }

  function shiftDasTimers(delta) {
    das.start += delta;
    if (das.charged) das.next += delta;
  }

  function togglePause() {
    if (state.over) return;
    paused = !paused;
    if (paused) {
      pausedAt = now();
    } else {
      // Freeze local DAS/ARR timers across the pause.
      shiftDasTimers(now() - pausedAt);
    }
    el.overlayPause.classList.toggle("hidden", !paused);
  }

  function clearHeldState() {
    if (softHeld) {
      softHeld = false;
      // Never send keyboard commands in jev mode; just drop the local state.
      if (state.mode !== "jev") {
        sendCommand("soft_drop", { phase: "up" });
      }
    }
    das.dir = null;
    das.charged = false;
    das.leftHeld = false;
    das.rightHeld = false;
  }

  document.addEventListener("keydown", (e) => {
    if (!GAME_KEYS.has(e.code)) return;
    e.preventDefault();
    if (e.repeat) return; // browser auto-repeat — client implements its own DAS/ARR

    if (e.code === "KeyR") {
      if (state.over) location.reload();
      return;
    }
    if (e.code === "KeyP") {
      togglePause();
      return;
    }
    if (paused || state.over) return; // pause freezes all input except P/R
    // Jev mode: the LLM plays — the client must not send keyboard commands.
    // P/R above stay local (pause overlay / reload).
    if (state.mode === "jev") return;

    switch (e.code) {
      case "ArrowLeft":
        pressDir("left");
        break;
      case "ArrowRight":
        pressDir("right");
        break;
      case "ArrowDown":
        if (!softHeld) {
          softHeld = true;
          sendCommand("soft_drop", { phase: "down" });
        }
        break;
      case "ArrowUp":
      case "KeyX":
        sendCommand("rotate_cw");
        break;
      case "KeyZ":
        sendCommand("rotate_ccw");
        break;
      case "KeyC":
      case "ShiftLeft":
      case "ShiftRight":
        sendCommand("hold");
        break;
      case "Space":
        sendCommand("hard_drop");
        break;
      default:
        break;
    }
  });

  document.addEventListener("keyup", (e) => {
    switch (e.code) {
      case "ArrowLeft":
        releaseDir("left");
        break;
      case "ArrowRight":
        releaseDir("right");
        break;
      case "ArrowDown":
        // Always process release (even during pause) so held state
        // cannot desync; only send if a "down" was actually sent.
        if (softHeld) {
          softHeld = false;
          sendCommand("soft_drop", { phase: "up" });
        }
        break;
      default:
        break;
    }
  });

  // If the window loses focus, keys can get stuck; clear held state.
  window.addEventListener("blur", () => {
    if (softHeld) {
      softHeld = false;
      sendCommand("soft_drop", { phase: "up" });
    }
    das.dir = null;
    das.charged = false;
    das.leftHeld = false;
    das.rightHeld = false;
  });

  // Per-frame DAS/ARR: one move command per ARR tick while charged.
  function updateDas(t) {
    if (paused || state.over || state.mode === "jev" || !das.dir) return;
    if (!das.charged) {
      if (t - das.start < DAS_MS) return;
      das.charged = true;
      das.next = das.start + DAS_MS;
    }
    // Catch up on missed ARR ticks (frame drops), bounded to avoid bursts.
    let guard = 0;
    while (t >= das.next && guard < 10) {
      sendCommand(das.dir === "left" ? "move_left" : "move_right");
      das.next += ARR_MS;
      guard += 1;
    }
  }

  // ------------------------------------------------------------------
  // Rendering
  // ------------------------------------------------------------------

  function drawCellAt(ctx, px, py, size, color) {
    ctx.fillStyle = color;
    ctx.fillRect(px + 1, py + 1, size - 2, size - 2);
    // Subtle top-left bevel for depth.
    ctx.fillStyle = "rgba(255,255,255,0.14)";
    ctx.fillRect(px + 1, py + 1, size - 2, 3);
  }

  function drawCell(ctx, col, row, color) {
    drawCellAt(ctx, col * CELL, row * CELL, CELL, color);
  }

  function renderBoard() {
    const W = COLS * CELL;
    const H = ROWS * CELL;
    bctx.fillStyle = "#0d1117";
    bctx.fillRect(0, 0, W, H);

    bctx.strokeStyle = "rgba(255,255,255,0.05)";
    bctx.lineWidth = 1;
    bctx.beginPath();
    for (let c = 1; c < COLS; c += 1) {
      bctx.moveTo(c * CELL + 0.5, 0);
      bctx.lineTo(c * CELL + 0.5, H);
    }
    for (let r = 1; r < ROWS; r += 1) {
      bctx.moveTo(0, r * CELL + 0.5);
      bctx.lineTo(W, r * CELL + 0.5);
    }
    bctx.stroke();

    const board = Array.isArray(state.board) ? state.board : [];
    for (let r = 0; r < ROWS; r += 1) {
      const row = board[r];
      if (typeof row !== "string") continue;
      for (let c = 0; c < COLS; c += 1) {
        const ch = row[c];
        if (ch && ch !== ".") {
          drawCell(bctx, c, r, COLORS[ch] || "#94a3b8");
        }
      }
    }

    // Optional ghost piece — jev mode target landing placement, drawn at
    // ~35% alpha so it stays clearly distinguishable from the falling piece
    // (which is drawn on top of it below). y may be negative while the
    // target sits in hidden rows; off-board cells are simply skipped.
    const ghost = parsePiece(state.ghost);
    if (ghost) {
      const cells = ROTATIONS[ghost.type][ghost.rot];
      bctx.globalAlpha = 0.35;
      for (const [r, c] of cells) {
        const x = ghost.x + c;
        const y = ghost.y + r;
        if (x >= 0 && x < COLS && y >= 0 && y < ROWS) {
          drawCell(bctx, x, y, COLORS[ghost.type] || "#94a3b8");
        }
      }
      bctx.globalAlpha = 1;
    }

    // Optional falling piece — only if the server includes it.
    const piece = parsePiece(state.piece);
    if (piece) {
      const cells = ROTATIONS[piece.type][piece.rot];
      bctx.globalAlpha = 0.85;
      for (const [r, c] of cells) {
        const x = piece.x + c;
        const y = piece.y + r;
        if (x >= 0 && x < COLS && y >= 0 && y < ROWS) {
          drawCell(bctx, x, y, COLORS[piece.type]);
        }
      }
      bctx.globalAlpha = 1;
    }
  }

  /*
   * Tolerant parser for the optional "piece" field. Accepts a flat object
   * with the piece type under {type, kind, piece, name}, position under
   * {x, col, px} / {y, row, py} and rotation under {rot, rotation, r, state}.
   * A bare string (type without position) cannot be placed, so it is ignored.
   */
  function parsePiece(p) {
    if (!p || typeof p !== "object") return null;
    let type = p.type !== undefined ? p.type : p.kind !== undefined ? p.kind
      : p.piece !== undefined ? p.piece : p.name;
    if (typeof type !== "string") return null;
    type = type.toUpperCase();
    if (!(type in ROTATIONS)) return null;

    const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
    const x = num(p.x !== undefined ? p.x : p.col !== undefined ? p.col : p.px);
    const y = num(p.y !== undefined ? p.y : p.row !== undefined ? p.row : p.py);
    if (x === null || y === null) return null;

    let rot = num(p.rot !== undefined ? p.rot : p.rotation !== undefined ? p.rotation
      : p.state !== undefined ? p.state : p.r) || 0;
    rot = ((Math.round(rot) % 4) + 4) % 4;

    return { type, x: Math.round(x), y: Math.round(y), rot };
  }

  function drawMiniPiece(ctx, type, centerX, centerY, cell) {
    const cells = ROTATIONS[type][0];
    let minR = Infinity, maxR = -Infinity, minC = Infinity, maxC = -Infinity;
    for (const [r, c] of cells) {
      if (r < minR) minR = r;
      if (r > maxR) maxR = r;
      if (c < minC) minC = c;
      if (c > maxC) maxC = c;
    }
    const w = (maxC - minC + 1) * cell;
    const h = (maxR - minR + 1) * cell;
    const ox = centerX - w / 2 - minC * cell;
    const oy = centerY - h / 2 - minR * cell;
    for (const [r, c] of cells) {
      drawCellAt(ctx, ox + c * cell, oy + r * cell, cell, COLORS[type]);
    }
  }

  function renderHold() {
    const W = holdCanvas.width;
    const H = holdCanvas.height;
    hctx.fillStyle = "#0d1117";
    hctx.fillRect(0, 0, W, H);
    if (state.hold && COLORS[state.hold]) {
      drawMiniPiece(hctx, state.hold, W / 2, H / 2, 22);
    }
  }

  function renderNext() {
    const W = nextCanvas.width;
    const H = nextCanvas.height;
    nctx.fillStyle = "#0d1117";
    nctx.fillRect(0, 0, W, H);
    const slotH = H / 5;
    for (let i = 0; i < Math.min(5, state.next.length); i += 1) {
      const type = state.next[i];
      if (typeof type === "string" && COLORS[type]) {
        drawMiniPiece(nctx, type, W / 2, slotH * (i + 0.5), i === 0 ? 22 : 18);
      }
    }
  }

  function updateReadouts() {
    el.score.textContent = String(state.score);
    el.lines.textContent = String(state.lines);
    el.level.textContent = String(state.level);
    el.combo.textContent = String(state.combo);
    el.combo.classList.toggle("stat-muted", state.combo === 0);
    el.b2b.textContent = state.b2b ? "B2B" : "OFF";
    el.b2b.classList.toggle("b2b-on", state.b2b);
    el.b2b.classList.toggle("b2b-off", !state.b2b);
  }

  function render() {
    renderBoard();
    if (state.dirty) {
      renderHold();
      renderNext();
      updateReadouts();
      state.dirty = false;
    }
  }

  // ------------------------------------------------------------------
  // Frame loop (rAF, capped to ~60 fps)
  // ------------------------------------------------------------------

  let lastFrame = 0;
  function frame(t) {
    requestAnimationFrame(frame);
    if (t - lastFrame < FRAME_MS - 2) return; // skip frames on >60Hz displays
    lastFrame = t;
    updateDas(t);
    render();
  }

  setStatus("disconnected");
  connect();
  requestAnimationFrame(frame);
})();
