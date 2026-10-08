"""The errors the engine process and the server raise for a request."""
from __future__ import annotations



class EngineDied(RuntimeError):
    """The engine process ended in the middle of a request (on Linux, usually the out-of-memory killer)."""


class EngineStartError(EngineDied):
    """The engine ended before READY; the message is its own `basalt error:` line(s), else its last log line.
    `log` is the engine log's path, if any.  An EngineDied: a request that triggered the start answers 503."""

    def __init__(self, message: str, log: str | None = None):
        super().__init__(message)
        self.log = log


class EngineStarting(RuntimeError):
    """The engine is (re)starting and has not said READY yet: no context size to plan a request with - a 503,
    not a 400 about the prompt."""


class ModelBusy(RuntimeError):
    """Explicit model controls must not interrupt active or queued requests."""


class EngineSilent(EngineDied):
    """The engine said nothing for too long during a request (or never acknowledged a STOP): the two sides lost
    step - the engine waiting for its next command, the server for this request's end - and the server ended it.  An
    EngineDied, so the request ends with an error and the next one starts the engine again."""


class EngineStuck(RuntimeError):
    """The engine process did not end after QUIT, terminate and kill: the server keeps it (and says so) rather than
    reporting its GPU and RAM as given back."""


class GpuBusy(RuntimeError):
    """The model is unloaded and the GPU has less free VRAM than min_free_vram_mib: something else (a game, another
    model server) is using it, so the engine is not started into the little that is left."""
