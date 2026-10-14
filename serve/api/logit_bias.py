"""OpenAI `logit_bias`: a {token id: bias} map added to the logits before sampling, checked before any streaming."""
from __future__ import annotations

import math
import re

# the engine-side key of a request; the handler stores the checked map there, so only Chat Completions sets it
REQUEST_KEY = "_logit_bias"


def normalize(value, vocab_size: int | None = None) -> dict[int, float]:
    """The request's `logit_bias` as {token id: bias}; ValueError naming the problem otherwise.  Biases are -100..100
    (-100 bans the token); an empty or absent map is no bias."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("logit_bias: expected an object mapping token ids to biases")
    out: dict[int, float] = {}
    for key, bias in value.items():
        if isinstance(key, bool) or not isinstance(key, (str, int)) or not re.fullmatch(r"[0-9]+", str(key)):
            raise ValueError(f"logit_bias: {key!r} is not a token id (a non-negative whole number)")
        token = int(key)
        if token > 2**31 - 1 or (vocab_size is not None and token >= vocab_size):
            raise ValueError(f"logit_bias: token id {token} is outside the model's vocabulary")
        if token in out:
            raise ValueError(f"logit_bias: token id {token} appears twice")
        if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not -100 <= bias <= 100 \
                or not math.isfinite(bias):
            raise ValueError(f"logit_bias: the bias of token {token} must be a number from -100 to 100")
        out[token] = float(bias)
    if vocab_size is not None and sum(b == -100 for b in out.values()) >= vocab_size:
        raise ValueError("logit_bias: it bans every token")
    return out


def engine_key(bias: dict[int, float] | None) -> str:
    """The engine protocol's ` logit_bias=id:value,...` (sorted by id), or nothing.  Only a map with integer ids, as
    `normalize` makes it, is sent: a request body (whose JSON keys are strings) cannot carry one."""
    if not bias or not all(isinstance(i, int) and isinstance(b, float) for i, b in bias.items()):
        return ""
    return " logit_bias=" + ",".join(f"{i}:{b:g}" for i, b in sorted(bias.items()))


def checked(req: dict, engine) -> dict[int, float]:
    """The request's bias, refused when this engine cannot apply it.  The engine reports its vocabulary size as
    INFO logit_bias=N (absent from a started engine: no support).  An engine that has not started (lazy load) or has
    no `info` (the mock) is not judged."""
    info = getattr(engine, "info", None)
    started = isinstance(info, dict) and "context" in info
    vocab = info.get("logit_bias") if started else None
    known = isinstance(vocab, int) and vocab > 1
    bias = normalize(req.get("logit_bias"), vocab if known else None)
    if bias and started and not known:
        raise ValueError("logit_bias: this engine build cannot apply token biases")
    if bias and getattr(engine, "batch", 0):
        raise ValueError("logit_bias: not supported with parallel slots (--parallel)")
    return bias
