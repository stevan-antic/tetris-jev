"""Jev (TypeSafe AI) choice-API client + env.yaml config loader.

env.yaml is NOT valid YAML (its profiles block uses malformed entries), so
the config is read with a tolerant line-based parser scoped to the
``typesafe_classifier:`` block. The api_key is loaded at runtime only and
is never committed, logged, or written into any test.

The HTTP touchpoint is ``_post`` — tests monkeypatch it, so no test ever
touches the network.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "env.yaml"

# Block header: exactly "typesafe_classifier" + colon (distinguishes it from
# the malformed "typesafe_classifier_jev" profile entries elsewhere).
_BLOCK_RE = re.compile(r"^(\s*)typesafe_classifier:\s*$")
# Indented "key: value" with an optional trailing "# comment".
_KV_RE = re.compile(r"^(\s+)([A-Za-z_][A-Za-z0-9_]*):\s*([^\s#]+)")


def load_jev_config(path: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Parse the typesafe_classifier block from env.yaml. Never raises;
    returns None when the file/block is missing or incomplete."""
    cfg: Dict[str, str] = {}
    try:
        lines = Path(path or _DEFAULT_PATH).read_text().splitlines()
    except OSError:
        return None
    in_block = False
    block_indent = 0
    for line in lines:
        m = _BLOCK_RE.match(line)
        if m:
            in_block = True
            block_indent = len(m.group(1))
            continue
        if not in_block:
            continue
        if not line.strip():
            continue
        km = _KV_RE.match(line)
        if km and len(km.group(1)) > block_indent:
            cfg[km.group(2)] = km.group(3)
        else:
            break  # dedent or non-key line ends the block
    if not cfg.get("api_key") or not cfg.get("base_url"):
        return None
    cfg.setdefault("endpoint", "/v1/systemone")
    cfg.setdefault("model", "jev-latest")
    return cfg


def make_question(state: dict, options) -> str:
    """One question: the goal plus the full option list (id + lines cleared +
    tspin + 20-row preview, rows joined with '/'). ~40 options x ~220 chars
    keeps the payload well under Jev's 64k context."""
    parts = [
        "You are playing Tetris. Goal: maximize lines cleared; avoid holes and uneven stack height.",
        f"Current board (before this piece is placed): piece={state.get('piece')} "
        f"hold={state.get('hold')} next={''.join(state.get('next') or [])} "
        f"level={state.get('level')} lines={state.get('lines')}.",
        "Choose ONE option. Each option shows the 20-row board AFTER the piece locks there.",
        "Options:",
    ]
    for o in options:
        parts.append(
            f"{o.id}: clears={o.lines_cleared} tspin={'yes' if o.tspin else 'no'} "
            f"board={'/'.join(o.preview)}"
        )
    parts.append("Reply with the id of the best option.")
    return "\n".join(parts)


def _norm(value: Any) -> Optional[str]:
    """Coerce a response value to an option-id string, defensively."""
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, dict):
        for k in ("id", "option", "option_id", "choice", "value", "answer", "name"):
            v = value.get(k)
            if isinstance(v, str):
                return v
            if isinstance(v, int) and not isinstance(v, bool):
                return str(v)
    return None


def extract_choice(resp: Any) -> Tuple[Optional[str], Optional[dict]]:
    """Pull (option_id, probabilities) out of a response of unknown shape.
    Checked in order: top-level choice/answer/selection/chosen/decision/
    option/option_id, then the first element of answers/choices (including
    OpenAI-style messages[].content). Probabilities from
    probabilities/probs/scores when present as a dict. (None, None) if
    nothing recognizable is found."""
    if not isinstance(resp, dict):
        return None, None
    probs = None
    for pk in ("probabilities", "probs", "scores"):
        p = resp.get(pk)
        if isinstance(p, dict):
            probs = p
            break
    val = None
    for key in ("choice", "answer", "selection", "chosen", "decision",
                "option", "option_id"):
        if key in resp:
            val = _norm(resp[key])
            if val is not None:
                break
    if val is None:
        for list_key in ("answers", "choices"):
            lst = resp.get(list_key)
            if isinstance(lst, list) and lst:
                first = lst[0]
                if isinstance(first, dict):
                    for k in ("id", "option", "option_id", "choice", "value",
                              "answer", "text"):
                        if k in first:
                            val = _norm(first[k])
                            if val is not None:
                                break
                    if val is None and isinstance(first.get("message"), dict):
                        val = _norm(first["message"].get("content"))
                else:
                    val = _norm(first)
                if val is not None:
                    break
    return val, probs


async def _post(url: str, headers: Dict[str, str], payload: dict,
                timeout_s: float) -> Any:
    """The single HTTP touchpoint — monkeypatch this in tests."""
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()


async def call(config: Dict[str, str], state: dict, options) \
        -> Tuple[Optional[str], Optional[dict]]:
    """One Jev round trip: POST {state, model, questions} with Bearer auth.
    Raises on timeout/HTTP errors — callers treat that as the fallback path."""
    payload = {
        "state": state,
        "model": config["model"],
        "questions": [make_question(state, options)],
    }
    url = config["base_url"].rstrip("/") + config.get("endpoint", "/v1/systemone")
    headers = {"Authorization": f"Bearer {config['api_key']}"}
    timeout_s = float(config.get("timeout_seconds", 30))
    resp = await _post(url, headers, payload, timeout_s)
    return extract_choice(resp)


class JevBrain:
    """Default brain: real Jev API. Any failure (missing config, timeout,
    HTTP error, unparseable response) yields (None, None) — the caller then
    applies the heuristic fallback, so the game never stalls."""

    def __init__(self, config: Optional[Dict[str, str]] = None) -> None:
        self._config = config

    def _cfg(self) -> Dict[str, str]:
        if self._config is None:
            self._config = load_jev_config() or {}
        return self._config

    async def choose(self, state: dict, options) \
            -> Tuple[Optional[str], Optional[dict]]:
        cfg = self._cfg()
        if not cfg.get("api_key") or not cfg.get("base_url"):
            return None, None
        try:
            return await call(cfg, state, options)
        except Exception:
            return None, None
