"""The Engine protocol and MockEngine, a scripted engine that needs no GPU."""
from __future__ import annotations

import threading
import time
from typing import Iterator, Protocol

from serve.util.text import IM_END


class Engine(Protocol):
    max_context: int
    def generate(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event) -> Iterator[int]: ...


class MockEngine:
    """Replays a scripted completion (text) as token ids, one per step, then the end-of-turn token.  Given a list of
    scripts, each request gets the next one and the last one repeats (a tool call, then the answer after it)."""

    def __init__(self, tokenizer, script: str | list[str], max_context: int = 32768, delay_s: float = 0.0):
        self.tok, self.max_context, self.delay = tokenizer, max_context, delay_s
        end = tokenizer.encode(IM_END, parse_special=True)
        self.scripts = [tokenizer.encode(x, parse_special=True) + end for x in ([script] if isinstance(script, str)
                                                                                 else script)]
        self.script, self.turns = self.scripts[0], 0
        self.last_prompt: list[int] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        self.last_embeddings = embeddings
        if len(self.scripts) > 1:
            self.script = self.scripts[min(self.turns, len(self.scripts) - 1)]
            self.turns += 1
        for t in self.script[:max_new]:
            if cancel.is_set():
                return
            if self.delay:
                time.sleep(self.delay)
            yield t
