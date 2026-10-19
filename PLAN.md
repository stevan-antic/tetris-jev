# Tetris Web App — Plan

Python backend (game engine + WebSocket server) and a web UI frontend. Two player modes share one command interface: human (keyboard/mouse) and LLM (OpenRouter choice API / plain LLM). Human mode ships first.

## Core design decisions

1. **Fixed-step simulation loop (60Hz).** The engine steps at a fixed rate; gravity is an independent accumulator timer scaled by level speed. Input is NEVER coupled to gravity ticks — a command applies on the next simulation step.
2. **Shared command vocabulary.** Both players issue the same commands: `move_left`, `move_right`, `rotate_cw`, `rotate_ccw`, `soft_drop`, `hard_drop`, `hold`. Commands enter a per-tick input queue and are applied at simulation-step granularity.
3. **LLM decides per piece, not per tick.** One LLM call per piece. The server enumerates all legal placements for the current piece (each rotation × each column, plus hold-then-place options) and offers them as choice-API options. The chosen option is translated by the server into the equivalent command sequence (`rotate ×n`, `move_left/right ×n`, `hard_drop`) fed into the same command interface a human uses. This keeps the interface shared while respecting LLM latency (500ms–3s per OpenRouter/Jev round trip) and giving Jev meaningful probabilities over a finite option list. A plain-LLM fallback (JSON answer over chat completions) uses the same option list.
4. **Client-side DAS/ARR.** The browser handles key auto-repeat with its own timers and sends press/release move intents over the WebSocket — raw OS key repeat is unreliable. The engine still enforces collision truth.
5. **Full guideline ruleset in phase 1:** SRS rotation with wall kicks, lock delay (500ms with a 15-move reset cap), spawn delay (ARE), input buffering across line-clear delays, hold piece, T-spin detection (minis included), combo, back-to-back scoring, 7-bag randomizer with a visible queue (hold + next 5).

## Speed levels (10 levels)

Gravity per level (ms per row), guideline-style curve:

| Level | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ms/row | 1000 | 793 | 618 | 473 | 355 | 262 | 190 | 135 | 94 | 64 |

Level up every 10 lines. Hard drop = instant lock (after lock-delay reset rules); soft drop = 20× gravity.

## Architecture

```
tetris-jev/
├── backend/
│   ├── main.py            # FastAPI app, serves static frontend + WS endpoint
│   ├── engine/
│   │   ├── board.py       # board state, collision, line clears
│   │   ├── pieces.py      # tetromino defs, SRS rotation states, wall kick tables
│   │   ├── game.py        # fixed-step loop, gravity/lock/spawn timers, scoring
│   │   └── rng.py         # 7-bag
│   ├── players/
│   │   ├── base.py        # PlayerController interface: submit(command)
│   │   ├── human.py       # WS command relay
│   │   └── llm.py         # placement enumeration + OpenRouter choice client
│   ├── protocol.py        # snapshot serialization, message types, seq numbers
│   └── bench.py           # headless harness: N games per brain, metrics
├── frontend/
│   ├── index.html
│   ├── game.js            # canvas render, keyboard input, DAS/ARR, WS client
│   └── style.css
└── tests/
    ├── test_engine.py     # collision, rotation+kicks, lock delay, T-spins, scoring
    ├── test_protocol.py   # seq/resync round-trip
    └── test_llm_stub.py   # deterministic stub brain against real engine
```

Stack: Python 3.12+, FastAPI, uvicorn (single WS + static server), pytest. Frontend: vanilla JS + Canvas, no build step. LLM calls via OpenRouter REST (Jev choice API when the guide's endpoint is available; otherwise `response_format: json` chat completions).

## WebSocket protocol (phase 1, not polish)

- Every state snapshot carries a monotonically increasing `seq`.
- Client acks latest seq; on gap detection client sends `resync` and server replies with a full-state snapshot.
- Message types defined up front: `snapshot` (board as 20 rows of single-char strings: `.` empty, `IJLOSTZ` locked), `game_over`, `ack`, `resync`, `command` (inbound), `option_request`/`option_choice` (LLM mode only).
- Command messages carry an optional client timestamp; server buffers commands arriving between sim steps and applies them on the next step (input buffering).

## Phases

### Phase 1 — Engine + human mode
1. Engine core: board, collision, 7-bag, SRS rotation + wall kicks, gravity accumulator, lock delay + move reset cap, spawn delay, line clears with scoring (single 100 / double 300 / triple 500 / tetris 800, T-spin classes, combo, B2B ×1.5).
2. Fixed-step loop (60Hz) with per-level gravity table, soft/hard drop, hold.
3. WebSocket server: snapshots with seq/ack/resync, command ingestion with buffering.
4. Frontend: canvas board, hold + next-5 queue, score/lines/level, keyboard controls (arrows move/rotate, Z/CCW, C hold, space hard drop, shift… finalize mapping), client-side DAS/ARR, reconnect with resync.
5. Tests: engine unit tests (kicks, T-spin detection, scoring, lock delay), protocol resync test.
6. Acceptance: play a full game at levels 1–5 in the browser; deliberate DAS taps, hold, and T-spin attempts behave correctly; kill the WS mid-game and see the client resume.

### Phase 2 — LLM mode
1. `players/llm.py`: enumerate legal placements each new piece (rotation × column × hold-variant), serialize each with resulting board preview + lines cleared.
2. Jev choice-API client with plain-LLM JSON fallback; one call per piece; timeout → server picks the placement with best simple heuristic (so the game never stalls).
3. UI toggle: human/LLM mode; LLM decisions are visible as ghost-piece highlights of considered options (optional, cheap).
4. Bench harness (`bench.py`): headless, run N games with a stub brain and a recorded decision log replay mode; metrics: pieces placed, lines, max level, invalid-choice rate.
5. Acceptance: stub brain completes games headlessly; a real LLM via OpenRouter survives level 3+ for 50+ pieces without stalling the loop; bench report produced.

### Phase 3 — Polish
Game over/restart flow, high-score persistence (SQLite), settings (starting level, ghost piece), LLM game spectating/replay viewer.

## Risks

- **LLM latency at high levels:** mitigated by per-piece decisions and a heuristic fallback when a choice times out; at level 8+ gravity (135ms/row) the LLM plays with lock-delay buffer, which is acceptable — target playability through level ~6.
- **LLM spatial reasoning:** mitigated by enumerating concrete placements with board previews so the model evaluates outcomes instead of simulating gravity/rotation mentally.
- **Token cost:** ~40 options × ~100 tokens per piece decision; per-piece (not per-tick) calls keep this bounded.
- **Latency spikes on OpenRouter:** per-request timeout with fallback keeps the game loop running.
