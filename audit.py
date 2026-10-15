"""Append only audit trail.

Every clinical request is logged with a request id, endpoint, model and prompt
versions, a hash of the redacted input and summary outcome metrics. Raw inputs
are never stored, which keeps the log useful for governance (who asked what
kind of question, with which model version, and how grounded was the answer)
without retaining patient information.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT,
    endpoint TEXT NOT NULL,
    input_sha256 TEXT,
    mode TEXT,
    model TEXT,
    prompt_version TEXT,
    latency_ms INTEGER,
    outcome TEXT
);
"""


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with sqlite3.connect(path) as con:
            con.executescript(SCHEMA)

    def record(self, endpoint: str, redacted_input: str, request_id: str | None = None, mode: str | None = None,
               model: str | None = None, prompt_version: str | None = None, latency_ms: int | None = None,
               outcome: dict | None = None) -> None:
        digest = hashlib.sha256(redacted_input.encode("utf-8")).hexdigest()
        with self._lock, sqlite3.connect(self.path) as con:
            con.execute(
                "INSERT INTO audit (ts, request_id, endpoint, input_sha256, mode, model, prompt_version, latency_ms, outcome) VALUES (?,?,?,?,?,?,?,?,?)",
                (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), request_id, endpoint, digest, mode, model, prompt_version, latency_ms, json.dumps(outcome or {})),
            )

    def recent(self, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["outcome"] = json.loads(d["outcome"] or "{}")
            out.append(d)
        return out

    def stats(self) -> dict:
        with sqlite3.connect(self.path) as con:
            total = con.execute("SELECT COUNT(*) FROM audit").fetchone()[0]
            by_endpoint = dict(con.execute("SELECT endpoint, COUNT(*) FROM audit GROUP BY endpoint").fetchall())
            avg_latency = con.execute("SELECT AVG(latency_ms) FROM audit WHERE latency_ms IS NOT NULL").fetchone()[0]
        return {"total_requests": total, "by_endpoint": by_endpoint, "mean_latency_ms": round(avg_latency or 0, 1)}
