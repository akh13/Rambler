# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional: keep an Open WebUI knowledge base up to date with your transcripts.

Each recording becomes one text file in the knowledge base (summary and speakers on top, then the
transcript). When a transcript changes, the old file is removed from the knowledge base and the new
one added. What's been sent is remembered in ~/.local/share/rambler/openwebui.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from .config import DATA_DIR, Config
from .transcript import Transcript, read_summary

STATE = DATA_DIR / "openwebui.json"


class OpenWebUIError(RuntimeError):
    pass


class Client:
    def __init__(self, url: str, key: str, timeout: float = 60):
        self.url = url.rstrip("/")
        self.key = key
        self.timeout = timeout

    def _req(self, method: str, path: str, body: bytes | None = None, ctype: str | None = None):
        req = urllib.request.Request(self.url + path, data=body, method=method)
        req.add_header("Authorization", f"Bearer {self.key}")
        req.add_header("Accept", "application/json")
        if ctype:
            req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise OpenWebUIError(f"{method} {path}: HTTP {e.code} {detail}") from None
        except (urllib.error.URLError, OSError) as e:
            raise OpenWebUIError(f"Can't reach Open WebUI at {self.url}: {e}") from None
        try:
            return json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return None

    def _json(self, method: str, path: str, payload) -> dict:
        return self._req(method, path, json.dumps(payload).encode(), "application/json")

    def knowledge_bases(self) -> list[tuple[str, str]]:
        """[(id, name)] the key can see."""
        data = self._req("GET", "/api/v1/knowledge/list")
        if isinstance(data, dict):        # newer versions page their results
            data = data.get("items") or data.get("data") or []
        return [(k["id"], k.get("name") or k["id"]) for k in data or [] if isinstance(k, dict) and "id" in k]

    def upload(self, name: str, text: str) -> str:
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                f"Content-Type: text/plain; charset=utf-8\r\n\r\n").encode() + text.encode() + \
               f"\r\n--{boundary}--\r\n".encode()
        data = self._req("POST", "/api/v1/files/?process=true", body, f"multipart/form-data; boundary={boundary}")
        if not isinstance(data, dict) or "id" not in data:
            raise OpenWebUIError(f"Upload of {name} returned no file id: {data!r}"[:300])
        return data["id"]

    def add(self, kb: str, file_id: str, tries: int = 6) -> None:
        # The server may still be processing the upload; it says so with a 400. Give it a moment.
        for i in range(tries):
            try:
                self._json("POST", f"/api/v1/knowledge/{kb}/file/add", {"file_id": file_id})
                return
            except OpenWebUIError as e:
                if i == tries - 1 or "HTTP 400" not in str(e):
                    raise
                time.sleep(1.5 * (i + 1))

    def remove(self, kb: str, file_id: str) -> None:
        try:
            self._json("POST", f"/api/v1/knowledge/{kb}/file/remove", {"file_id": file_id})
        except OpenWebUIError as e:
            if "HTTP 404" not in str(e) and "HTTP 400" not in str(e):
                raise


def document(rec) -> str:
    t = Transcript.for_recording(rec)
    head = [f"Recording: {rec.rel.as_posix()}"]
    summ = read_summary(rec)
    if summ:
        head.append(f"Summary: {summ}")
    if t.speakers():
        head.append("Speakers: " + ", ".join(t.speakers()))
    if t.segments and any(s.speaker for s in t.segments):
        lines, last = [], None
        for s in t.segments:
            if s.speaker and s.speaker != last:
                lines.append(f"\n{s.speaker}:")
                last = s.speaker
            lines.append(s.text)
        body = " ".join(lines).replace(" \n", "\n").strip()
    else:
        body = t.text().strip()
    return "\n".join(head) + "\n\n" + body + "\n"


def _load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(d: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, STATE)


def push(cfg: Config, recs, log=print, cancel=None) -> tuple[int, int]:
    """Send new or changed transcripts. Returns (sent, failed)."""
    if not (cfg.owui_url and cfg.owui_key and cfg.owui_knowledge):
        return 0, 0
    c = Client(cfg.owui_url, cfg.owui_key)
    state = _load_state()
    kb_state = state.setdefault(cfg.owui_knowledge, {})
    sent = failed = 0
    for r in recs:
        if cancel is not None and cancel.is_set():
            break
        if not r.txt:
            continue
        doc = document(r)
        sig = hashlib.sha1(doc.encode()).hexdigest()
        key = r.rel.as_posix()
        prev = kb_state.get(key)
        if prev and prev.get("sig") == sig:
            continue
        try:
            fid = c.upload(Path(r.stem).name + ".txt", doc)
            c.add(cfg.owui_knowledge, fid)
            if prev and prev.get("file_id"):
                c.remove(cfg.owui_knowledge, prev["file_id"])
            kb_state[key] = {"file_id": fid, "sig": sig}
            _save_state(state)
            sent += 1
            log(f"Open WebUI: {'updated' if prev else 'added'} {r.stem}")
        except OpenWebUIError as e:
            failed += 1
            log(f"Open WebUI: {r.stem}: {e}")
            if "Can't reach" in str(e):
                break
    return sent, failed
