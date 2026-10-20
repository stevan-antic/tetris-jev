"""Selectable brain profiles: env.yaml resolution + chat-model brains.

env.yaml is parsed with yaml.safe_load (it is valid YAML now). If the file
ever becomes invalid again, the typesafe block keeps working through the
tolerant line-based parser in jev_api.py — so the default Jev mode survives
a re-broken user config.

A profile is selectable for the player iff it maps (via ``llm:`` or
``classifier:``) to a defined backend of kind ``typesafe`` /
``openai_compatible`` / ``anthropic``. A profile whose backend is missing
(e.g. local-copilot -> copilot_bridge) stays listed but ``available=false``.

api keys live in env.yaml only: they are read at runtime and used inside
request headers; they are never hardcoded, logged, or written into tests.

The HTTP touchpoint is ``_post`` (shared shape with jev_api._post) — tests
monkeypatch it, so no test ever touches the network.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .jev_api import JevBrain, load_jev_config

_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "env.yaml"

# Backend kinds a selectable profile may resolve to. Unknown kinds (or a
# missing backend id) leave the profile listed but unavailable.
_SELECTABLE_KINDS = ("typesafe", "openai_compatible", "anthropic")

# Option-id vocabulary produced by llm.enumerate_placements, e.g. r1_c7 or
# hold_r0_c3. Matching is done over the raw reply text; word boundaries keep
# "r1_c7" from matching inside "r11_c7".
_OPTION_ID_RE = re.compile(r"\b(hold_)?r([0-3])_c(\d{1,2})\b")


def load_config(path: Optional[str] = None) -> Optional[dict]:
    """Parse env.yaml with yaml.safe_load; None when the file is missing or
    does not parse (callers then fall back to the tolerant typesafe parser).
    Secrets stay in the returned dict; this function never logs."""
    try:
        import yaml

        data = yaml.safe_load(Path(path or _DEFAULT_PATH).read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def resolve_profiles(path: Optional[str] = None) -> List[dict]:
    """The player-selectable profiles: [{"id", "available"}], sorted by id.

    A profile is selectable iff its backend (via llm:/classifier:) exists and
    is one of the selectable kinds. A profile that names an UNDEFINED backend
    (e.g. local-copilot -> copilot_bridge) is still listed, with
    available=false, so the UI can show it greyed out.
    """
    data = load_config(path)
    if data is None:
        return []
    profiles = data.get("profiles")
    backends = data.get("backends")
    if not isinstance(profiles, dict):
        return []
    # Backend definitions sit in the same ``profiles:`` map — an entry is a
    # backend definition iff it carries a ``kind:`` key.
    if not isinstance(backends, dict):
        backends = {k: v for k, v in profiles.items()
                    if isinstance(v, dict) and "kind" in v}
    out: List[dict] = []
    for pid, spec in profiles.items():
        if not isinstance(spec, dict) or "kind" in spec:
            continue  # backend definitions are not player-selectable
        backend_id = spec.get("llm") or spec.get("classifier")
        if not isinstance(backend_id, str):
            continue
        backend = backends.get(backend_id)
        available = isinstance(backend, dict) and backend.get("kind") in _SELECTABLE_KINDS
        out.append({"id": pid, "available": bool(available)})
    out.sort(key=lambda p: p["id"])
    return out


def default_profile_id(path: Optional[str] = None) -> Optional[str]:
    """env.yaml's ``default_profile:`` value when it is a selectable,
    available profile; None otherwise (caller falls back to the current
    default — the typesafe JevBrain)."""
    data = load_config(path)
    if data is None:
        return None
    want = data.get("default_profile")
    for p in resolve_profiles(path):
        if p["id"] == want:
            return want if p["available"] else None
    return None


def backend_for(profile_id: str, path: Optional[str] = None) -> Optional[dict]:
    """Resolve a profile id to its backend definition dict; None when the
    profile is unknown or its backend is missing."""
    data = load_config(path)
    if data is None:
        return None
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        return None
    spec = profiles.get(profile_id)
    if not isinstance(spec, dict) or "kind" in spec:
        return None
    backend_id = spec.get("llm") or spec.get("classifier")
    if not isinstance(backend_id, str):
        return None
    # Backend definitions live in the same map (see resolve_profiles).
    backends = {k: v for k, v in profiles.items()
                if isinstance(v, dict) and "kind" in v}
    backend = backends.get(backend_id)
    if not isinstance(backend, dict) or backend.get("kind") not in _SELECTABLE_KINDS:
        return None
    return backend


def build_brain(profile_id: str, path: Optional[str] = None):
    """Profile -> brain instance, or None when the profile is not resolvable
    to a selectable backend. kind=typesafe reuses JevBrain as-is (its tolerant
    parser works whether or not the rest of env.yaml parses); chat kinds get
    a ChatBrain."""
    backend = backend_for(profile_id, path)
    if backend is None:
        return None
    kind = backend.get("kind")
    if kind == "typesafe":
        # Prefer the tolerant parser for the typesafe block so Jev keeps
        # working even when the rest of env.yaml no longer parses.
        return JevBrain(config=load_jev_config(path))
    if kind in ("openai_compatible", "anthropic"):
        return ChatBrain(profile_id=profile_id, backend=backend)
    return None


def extract_option_id(text: Any) -> Optional[str]:
    """Pull an option id (r<rot>_c<col> / hold_r<rot>_c<col>) out of a chat
    reply. Returns the FIRST match, or None when the text holds no valid id.
    A match is only trusted if the referenced rotation is 0-3 (guaranteed by
    the regex); the column is not validated here — the caller looks the id up
    among the live options and falls back to the heuristic on a miss."""
    if not isinstance(text, str):
        return None
    m = _OPTION_ID_RE.search(text)
    if not m:
        return None
    return m.group(0)


def make_prompt(state: dict, options) -> str:
    """The ChatBrain question: compact board + piece context + the option
    list as 'id: preview', then the strict reply instruction."""
    parts = [
        "You are playing Tetris. Pick the best placement for the piece.",
        "Board is 10 wide, 20 rows top to bottom. Goal: clear lines, avoid holes, keep the stack low.",
        f"Piece={state.get('piece')} Hold={state.get('hold')} Next={''.join(state.get('next') or [])} "
        f"Level={state.get('level')} Lines={state.get('lines')}.",
        "Board BEFORE the drop:",
        "/".join(state.get("board") or []),
        "Options (id: board AFTER the piece locks there):",
    ]
    for o in options:
        parts.append(f"{o.id}: {'/'.join(o.preview)}")
    parts.append("Reply with ONLY the option id of the best move.")
    return "\n".join(parts)


async def _post(url: str, headers: Dict[str, str], payload: dict,
                timeout_s: float) -> Any:
    """The single HTTP touchpoint — monkeypatch this in tests."""
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()


class ChatBrain:
    """Brain for chat-style backends (kind openai_compatible / anthropic).

    choose(state, options) -> (option_id | None, None): one round trip, the
    reply parsed for an option id. Any failure (missing key, timeout, HTTP
    error, garbage reply) yields (None, None) so the session falls back to
    the heuristic — the game never stalls. Same contract as JevBrain."""

    def __init__(self, profile_id: str, backend: dict,
                 config_path: Optional[str] = None) -> None:
        self.profile_id = profile_id
        self._backend = dict(backend)
        self._config_path = config_path

    # -- request shaping ------------------------------------------------

    def _request(self, state: dict, options) -> Tuple[str, Dict[str, str], dict, float]:
        """(url, headers, payload, timeout_s) for one call. Raises KeyError on
        a backend definition missing its mandatory fields — the caller
        treats that as the fallback path."""
        kind = self._backend["kind"]
        prompt = make_prompt(state, options)
        timeout_s = float(self._backend.get("timeout_seconds", 30))
        if kind == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self._backend["api_key"],
                "anthropic-version": "2023-06-01",
            }
            payload = {
                "model": self._backend["model"],
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
            return url, headers, payload, timeout_s
        # openai_compatible
        base = self._backend["base_url"].rstrip("/")
        if not base.endswith("/chat/completions"):
            url = base + "/chat/completions"
        else:
            url = base
        headers = {}
        if self._backend.get("api_key"):
            headers["Authorization"] = f"Bearer {self._backend['api_key']}"
        payload = {
            "model": self._backend["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(512, int(self._backend.get("max_tokens", 512))),
            "temperature": 0,
        }
        return url, headers, payload, timeout_s

    @staticmethod
    def _reply_text(resp: Any, kind: str) -> Any:
        """The message text out of a chat-completions / messages response."""
        if not isinstance(resp, dict):
            return None
        if kind == "anthropic":
            content = resp.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    return first.get("text")
            return None
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    return msg.get("content")
        return None

    async def choose(self, state: dict, options) \
            -> Tuple[Optional[str], Optional[dict]]:
        try:
            url, headers, payload, timeout_s = self._request(state, options)
        except (KeyError, TypeError, ValueError):
            return None, None
        try:
            resp = await _post(url, headers, payload, timeout_s)
        except Exception:
            return None, None
        kind = self._backend.get("kind")
        text = self._reply_text(resp, kind)
        return extract_option_id(text), None
