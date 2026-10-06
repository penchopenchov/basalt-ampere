"""serve/logbuf.py - the last lines the server and the engine printed, for GET /api/logs.

`LogBuffer` is a ring of lines with rising ids; `TeeStream` feeds it what this process prints (without changing what
reaches the real stream); `tail_file` feeds it the engine's log file as it grows.
"""
from __future__ import annotations

import collections
import os
import re
import threading
import time

CAPACITY = 2000
MAX_LINE = 2000
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
ERR = re.compile(r"\b(?:error|failed|fatal|exception|traceback|died|abort)\b", re.I)
WARN = re.compile(r"\b(?:warn|warning|stuck|timeout|retry|retrying|slow)\b", re.I)
SERVER_PREFIX = "[server] "


def level(text: str) -> str:
    """"err", "warn" or "info", by the words in the line."""
    if ERR.search(text):
        return "err"
    return "warn" if WARN.search(text) else "info"


class LogBuffer:
    """A thread-safe ring of the last CAPACITY lines.  Ids start at 1 and only rise."""

    def __init__(self, capacity: int = CAPACITY):
        self.lock = threading.Lock()
        self.lines: collections.deque = collections.deque(maxlen=capacity)
        self.last = 0

    def add(self, src: str, text: str) -> None:
        """Adds each non-empty line of `text` (ANSI codes removed, long lines cut) from source `src`."""
        now = round(time.time(), 3)
        for raw in ANSI.sub("", text or "").splitlines():
            msg = raw.rstrip()[:MAX_LINE]
            if not msg.strip():
                continue
            with self.lock:
                self.last += 1
                self.lines.append({"id": self.last, "t": now, "src": src, "lvl": level(msg), "msg": msg})

    def since(self, after_id: int = 0, limit: int = 500) -> tuple[list[dict], int]:
        """-> (the newest `limit` lines with an id above `after_id`, oldest first; the newest id in the buffer)."""
        with self.lock:
            fresh = [dict(x) for x in self.lines if x["id"] > after_id]
            last = self.last
        return (fresh[-limit:] if limit > 0 else []), last


class TeeStream:
    """Wraps sys.stdout / sys.stderr: everything still goes to the original stream, and each complete line is also
    added to the buffer as `src`, without a leading "[server] "."""

    def __init__(self, stream, buffer: LogBuffer, src: str):
        self._stream, self._buffer, self._src = stream, buffer, src
        self._partial = ""
        self._lock = threading.Lock()

    def write(self, text):
        n = self._stream.write(text)
        if isinstance(text, str):
            with self._lock:
                self._partial += text
                *done, self._partial = self._partial.split("\n")
            for line in done:
                self._buffer.add(self._src, line[len(SERVER_PREFIX):] if line.startswith(SERVER_PREFIX) else line)
        return n

    def __getattr__(self, name):
        return getattr(self._stream, name)


def tail_file(buffer: LogBuffer, path: str, src: str = "eng", poll_s: float = 0.25) -> None:
    """Follows a text file from its current end, adding each complete line to the buffer.  A file that is not there
    yet is waited for; one that shrinks (emptied or replaced) is read again from its start.  Runs until the process
    ends."""
    try:
        pos = os.path.getsize(path)
    except OSError:
        pos = 0
    partial = b""
    while True:
        try:
            size = os.path.getsize(path)
            if size < pos:
                pos, partial = 0, b""
            if size > pos:
                with open(path, "rb") as f:
                    f.seek(pos)
                    data = f.read(min(size - pos, 1 << 20))
                pos += len(data)
                *done, partial = (partial + data).split(b"\n")
                for line in done:
                    buffer.add(src, line.decode("utf-8", "replace"))
                continue
        except OSError:
            pass
        time.sleep(poll_s)
