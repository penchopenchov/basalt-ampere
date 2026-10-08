"""Service: the engine behind a lock with its conversation cache, request bookkeeping and the metrics the stats page reads."""
from __future__ import annotations

import collections
import contextlib
import ctypes
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator

from serve.engine.errors import EngineDied, EngineSilent, EngineStartError, EngineStarting, EngineStuck, GpuBusy, ModelBusy
from serve.engine.mock import Engine
from serve.frontend import ChatTemplate, Event, OutputParser, images_of, literal_tags, mark_think_literals, unmark_think_literals
from serve.util.logbuf import LogBuffer
from serve.security import LOOPBACK_NAMES
from serve.util.text import Detokenizer, IM_END, StopMatcher, control_literals, stop_strings
from serve.engine.vision import Vision


IMAGE_PAD = "<|image_pad|>"
# the trailing effort turn ("effort_position": "end"); low is the template's own sentence, medium (which the
# template says nothing for: no xhigh sentence is medium there) says it, since the xhigh one stays at the top
EFFORT_TURN = "<|im_start|>system\n{}<|im_end|>\n"
EFFORT_TEXT = {"low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                      "conclusion without unnecessary elaboration.",
               "medium": "Reasoning effort is set to medium. Think as much as the task needs, without unnecessary "
                         "elaboration."}
VISION_START = "<|vision_start|>"
# a reply that repeats one token this many times in a row is ended there ("length"): a model in a loop, or a
# broken state that answers one token forever (an issue saw 36,689 tokens of "!"). The Service's
# repeat_stop_tokens sets it; 0 turns it off.
REPEAT_STOP_TOKENS = 256
# what closes the thinking when it reaches reasoning_budget_tokens (the model's own end-of-thinking tag after it)
REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n"
# what closes the thinking when the model ends its turn inside it, so it still answers
REASONING_CLOSE = "\n</think>\n\n"
CTX_SLACK = 8               # `basalt --serve` rejects prompt + max_new + 8 > context: keep the same margin here
# The live tok/s is a rate over a window, not a mean since the first token: a mean divides by a span near zero at
# the first token (a huge number) and then undershoots for the first second of every answer.
RATE_WINDOW_S = 2.0
LOG_FIRST_LINES = 300        # the lines GET /api/logs returns when the client has none yet
LOG_MAX_LINES = 500          # ... and the most it returns in one answer
REQUESTS_SHOWN = 12        # the finished requests GET /metrics lists, newest first
RATE_MIN_SPAN_S = 0.25      # younger than this there is no rate yet: the mean so far, with the span floored here
# Diagnostics, off unless the matching server flag is given (main sets them).
DEBUG = False            # --debug: one line per request, the raw model text after it


class Service:
    def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "qwen3.8-flash-next",
                 vision: Vision | None = None, sampling_defaults: dict | None = None,
                 fit_max_tokens: bool = False):
        self.engine, self.tok, self.template, self.model, self.vision = engine, tokenizer, template, model_name, vision
        self.literals = literal_tags(control_literals(tokenizer))   # texts that stay text inside a message
        self.fit_max_tokens = fit_max_tokens          # --fit-max-tokens: clamp the output cap instead of 400
        self.aliases: list[str] = []                  # other names of the model (--alias)
        self.sampling_defaults = dict(sampling_defaults or {})   # the model file's sampling block
        self.fifo = threading.Lock()
        self.embeddings = threading.local()           # the current request's image embeddings file (GENI)
        self.api_key = ""                              # when set, /v1/* needs it (Bearer or x-api-key)
        # a non-default reasoning effort goes in a short system
        # turn before the answer instead of the top of the prompt, so switching it keeps the cached conversation
        self.effort_end = False
        self.logs = LogBuffer()                         # GET /api/logs: the server's and the engine's output
        # browser pages of these origins may call /v1/* (CORS; "*" = any page - only with an api_key that
        # matters); empty = no CORS headers at all, as before
        self.cors_origins: list[str] = []
        # origins that count as the server's own page for the load/unload/VRAM controls, e.g. reached through a
        # reverse proxy or tunnel whose Host differs ("https://host.example.com"); never a wildcard
        self.trusted_origins: list[str] = []
        # DNS rebinding: extra Host names this server answers to (--allowed-host;
        # "*" = any), and every name it answers to, which serve() works out from the address it listens on
        self.allowed_hosts: list[str] = []
        self.host_names: set[str] = set(LOOPBACK_NAMES)
        self.status = {"busy": False, "queued": 0}      # GET /status: what the model is doing right now
        self.rate = collections.deque(maxlen=32)        # (time, generated) samples for the live tok/s window
        # "parallel" (the engine's batch slots): every running request's own status and rate window; self.status
        # then says busy while any runs and shows the newest one
        self.live_reqs: dict[int, tuple[dict, collections.deque]] = {}
        self.request_ctx = threading.local()            # the handler thread's request: .path (the API path serving it)
        self.history = collections.deque(maxlen=500)    # the last finished requests, newest last (GET /metrics)
        self.requests_done = 0                           # finished requests since the start (their ids, /v1/status)
        self.last_timings = None                         # the last finished request's, llama.cpp's names (/v1/status)
        self.last_request_at = None                      # when a request last started or finished
        self.started_at = time.time()
        self.status_lock = threading.Lock()
        # sharing the GPU with other programs (all off by default): unload the engine after this many idle seconds,
        # only start it again when this much VRAM is free, and run this command first (e.g. to unload another
        # server's model); the next request after an unload starts the engine again
        self.idle_unload_s = 0
        self.min_free_vram_mib = 0
        self.before_load = None
        self.vram_reserve = None                         # the last POST /v1/vram reserve (None: the start's)
        self.reasoning_budget_tokens = 0                 # a default thinking budget (0: none)
        self.repeat_stop_tokens = REPEAT_STOP_TOKENS     # one token this many times in a row ends a reply (0: off)
        self.anthropic_think_unasked = True               # "anthropic_thinking": "on_request" -> False
        self.stop_ids = set(tokenizer.encode(IM_END, parse_special=True) +
                            tokenizer.encode("<|endoftext|>", parse_special=True))

    def loaded(self) -> bool:
        return not hasattr(self.engine, "alive") or self.engine.alive()

    def set_aliases(self, aliases) -> None:
        """The model's other names - a list of names (or one comma-separated string), like llama-server's --alias.
        ValueError for anything else."""
        if aliases is None:
            aliases = []
        if isinstance(aliases, str):
            aliases = aliases.split(",")
        if not isinstance(aliases, list) or not all(isinstance(x, str) for x in aliases):
            raise ValueError(f"aliases: expected a list of names, not {aliases!r}")
        names = []
        for x in (x.strip() for x in aliases):
            if x and x != self.model and x not in names:
                names.append(x)
        self.aliases = names

    def model_names(self) -> list[str]:
        return [self.model, *self.aliases]

    def model_for(self, req) -> str:
        """The name to answer with: the request's own when it is the model's name or an alias, else the model's.
        Other names are still served, as before."""
        asked = req.get("model") if isinstance(req, dict) else None
        return asked if isinstance(asked, str) and asked in self.aliases else self.model

    def reasoning_budget(self, req) -> int | None:
        """The most tokens this request may think, or None: the request's `reasoning_budget_tokens`, else
        none.  0 (or less) means no budget.  ValueError (a 400) for
        anything that is not a whole number."""
        value = (req or {}).get("reasoning_budget_tokens") if isinstance(req, dict) else None
        if value is None:
            value = self.reasoning_budget_tokens
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"reasoning_budget_tokens: expected a whole number (0 = none), not {value!r}")
        return value if value > 0 else None

    def _vision_down(self) -> bool:
        return self.vision is not None and hasattr(self.vision, "alive") and not self.vision.alive()

    def free_vram_mib(self) -> int | None:
        """Free VRAM on the engine's (first) GPU, from NVML; None when it can't be read (then nothing is refused)."""
        try:
            from serve.util.telemetry import _Nvml
            nv = _Nvml(int(getattr(self, "gpu_index", 0) or 0))
            if not nv.ok():
                return None
            m = nv.Mem()
            if nv.lib.nvmlDeviceGetMemoryInfo(nv.dev, ctypes.byref(m)) != 0:
                return None
            return int(m.free >> 20)
        except Exception:
            return None

    def ensure_loaded(self):
        """Start the engine if it is not running (unloaded, or it died -), after the before_load hook and
        the free-VRAM check.  The caller holds self.fifo."""
        if self.loaded() and not self._vision_down():
            return
        if self.before_load:
            cmd = self.before_load
            print(f"[server] before_load: {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}", flush=True)
            try:
                subprocess.run(cmd, shell=isinstance(cmd, str), timeout=120, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError) as e:
                print(f"[server] warning: before_load command failed ({e}), loading anyway", flush=True)
        if self.min_free_vram_mib:
            free = self.free_vram_mib()
            deadline = time.time() + 15                 # memory another process just gave back can take a moment
            while free is not None and free < self.min_free_vram_mib and time.time() < deadline:
                time.sleep(0.5)
                free = self.free_vram_mib()
            if free is not None and free < self.min_free_vram_mib:
                raise GpuBusy(f"the GPU is in use by another program: {free} MiB of VRAM free, the model needs "
                              f"{self.min_free_vram_mib} (min_free_vram_mib) - it stays unloaded until that is free")
        if self._vision_down():                         # first, as at a start: a GPU encoder takes its VRAM before
            print("[server] restarting the vision encoder", flush=True)   # the engine sizes its cache
            self.vision.restart()
        if self.loaded():
            return
        if getattr(self.engine, "unloaded", False):
            print("[server] loading the model again", flush=True)
        else:
            code = self.engine.exit_code() if hasattr(self.engine, "exit_code") else None
            print(f"[server] warning: the engine stopped (exit code {code}), restarting it", flush=True)
        try:
            self.engine.restart()
        except EngineStartError as e:                   # a line, not a traceback: the next request tries again
            print(f"[server] error: {e}", flush=True)
            if e.log:
                print(f"[server] engine log: {e.log}", flush=True)
            raise
        print("[server] engine running again", flush=True)
        if self.vram_reserve is not None and hasattr(self.engine, "vram"):   # the reserve asked for last
            try:
                self.engine.vram(self.vram_reserve)
            except (ValueError, EngineDied) as e:
                print(f"[server] warning: VRAM reserve of {self.vram_reserve} MiB not applied ({e})", flush=True)

    vram_wait_s = 300.0                                  # how long POST /v1/vram waits for a running request

    def vram(self, reserve_mib: int | None) -> dict:
        """, POST /v1/vram: keep `reserve_mib` of VRAM free for other programs (None: the reserve the engine
        started with), applied between requests - a request that is running finishes first (up to wait_s).  Only an
        engine started with --vram-elastic  can do it; it never resizes on its own.
        An unloaded engine applies it when it loads.  -> {"status": ..., and the engine's figures}."""
        if not hasattr(self.engine, "vram"):
            raise ValueError("this engine cannot resize its VRAM use")
        if not self.fifo.acquire(timeout=self.vram_wait_s):
            raise ModelBusy("a request is still running; try again when it has finished")
        # parallel requests do not hold the fifo: the engine's control lines (a prompt being admitted) are taken too,
        # and the engine refuses a resize while a slot decodes
        ctl = getattr(self.engine, "ctl", None) if getattr(self.engine, "batch", 0) else None
        if ctl is not None and not ctl.acquire(timeout=self.vram_wait_s):
            self.fifo.release()
            raise ModelBusy("a request is still running; try again when it has finished")
        try:
            if not self.loaded():
                self.vram_reserve = reserve_mib
                return {"status": "not loaded", "reserve_mib": reserve_mib,
                        "note": "applied when the model loads"}
            out = self.engine.vram(reserve_mib)
            self.vram_reserve = reserve_mib
            print(f"[server] vram: {out.get('vram_free_mib')} MiB free, expert cache {out.get('expert_cache_mib')} of "
                  f"{out.get('expert_cache_full_mib')} MiB ({out.get('expert_slots')} experts)", flush=True)
            return {"status": "ok", **out}
        finally:
            if ctl is not None:
                ctl.release()
            self.fifo.release()

    def _say_died(self, e: Exception) -> None:
        """The server window's line for an engine that died (or was ended) in the middle of a request."""
        note = self.engine.death_note() if hasattr(self.engine, "death_note") else ""
        log = getattr(self.engine, "log_path", None)
        why = f"{e}: {note}" if note else str(e)
        print(f"[server] warning: {why} (the next request restarts it{', log ' + log if log else ''})", flush=True)

    def load(self):
        """POST /load and every generation request: start the engine now if it is unloaded (raises GpuBusy)."""
        # a request is on its way: the idle thread must not unload between this and the request's own start
        self.last_request_at = time.time()
        if self.loaded() and not self._vision_down():
            return
        with self.fifo:
            self.ensure_loaded()

    def unload(self, idle_for: float | None = None) -> str:
        """Stop the engine between requests: "unloaded", "not loaded", "busy" (a request is running or waiting, or
        with idle_for: one ran more recently than that) or "unsupported"."""
        if not hasattr(self.engine, "unload"):
            return "unsupported"
        if not self.fifo.acquire(blocking=False):
            return "busy"
        try:
            if not self.engine.alive():
                return "not loaded"
            with self.status_lock:
                if self.status.get("busy") or self.status.get("queued"):
                    return "busy"
            if idle_for is not None and time.time() - (self.last_request_at or self.started_at) < idle_for:
                return "busy"
            self.engine.unload()
            if self.vision is not None and hasattr(self.vision, "unload"):
                self.vision.unload()
            print(f"[server] model unloaded{f' after {idle_for:.0f} s idle' if idle_for else ''}, the next request loads it", flush=True)
            return "unloaded"
        finally:
            self.fifo.release()

    def start_idle_unload(self):
        if not self.idle_unload_s or not hasattr(self.engine, "unload"):
            return
        print(f"[server] model unloads after {self.idle_unload_s} s idle", flush=True)

        def loop():
            while True:
                time.sleep(max(1.0, min(30.0, self.idle_unload_s / 4)))
                try:
                    self.unload(idle_for=self.idle_unload_s)
                except EngineStuck as e:                # tried again at the next turn; the thread keeps running
                    print(f"[server] warning: idle unload failed ({e})", flush=True)
        threading.Thread(target=loop, daemon=True).start()

    def start_telemetry(self):
        """The hardware sampler behind GET /metrics (serve/util/telemetry.py), recording this server's speeds too."""
        if getattr(self, "telemetry", None) is None:
            from serve.util.telemetry import Telemetry
            self.telemetry = Telemetry(extra=lambda: {"tok_s": self._tok_s(),
                                                    "prefill_tok_s_mean": self._prefill_tok_s_mean()},
                                       gpu_indices=getattr(self, "gpu_indices", None) or [0])

    def _tok_s(self):
        """tok/s over the last RATE_WINDOW_S seconds.  Returns 0.0 while nothing is generating.  With parallel requests:
        the sum of theirs."""
        with self.status_lock:
            many = [(dict(st), list(rt)) for st, rt in self.live_reqs.values()]
        if many:
            return sum(self._rate_of(st, rt) for st, rt in many)
        with self.status_lock:
            s = dict(self.status)
            rate = list(self.rate)
        return self._rate_of(s, rate)

    @staticmethod
    def _rate_of(s, rate):
        if not s.get("busy") or not s.get("first_token"):
            return 0.0
        now = time.time()
        newest = rate[-1] if rate else None
        oldest = next(((t, g) for t, g in rate if now - t <= RATE_WINDOW_S), None)
        if newest and oldest and newest[0] - oldest[0] >= RATE_MIN_SPAN_S:
            return max(0.0, (newest[1] - oldest[1]) / (newest[0] - oldest[0]))
        return s["generated"] / max(RATE_MIN_SPAN_S, now - s["first_token"])

    def _tok_s_mean(self):
        """The whole-request mean since the first token (the old formula), kept so the two can be compared."""
        with self.status_lock:
            many = [dict(st) for st, _ in self.live_reqs.values()] or [dict(self.status)]
        return sum(s["generated"] / max(1e-6, time.time() - s["first_token"]) for s in many
                   if s.get("busy") and s.get("first_token"))

    def _prefill_tok_s_mean(self):
        """Engine-reported mean over newly read tokens, excluding the cached prefix."""
        with self.status_lock:
            reading = self.status.get("busy") and self.status.get("first_token") is None
        return getattr(self.engine, "prefill_tok_s_mean", None) if reading else 0.0

    def metrics(self) -> dict:
        """GET /metrics: what the stats page shows - the engine's facts, what it is doing, the last requests, the
        hardware and a minute of history for the two speed series."""
        with self.status_lock:
            s = dict(self.status)
            if self.live_reqs:                          # parallel requests: the newest one's status
                s = {**s, **dict(list(self.live_reqs.values())[-1][0]), "queued": s.get("queued", 0)}
            hist = list(self.history)
        busy = bool(s.get("busy"))
        if busy and s.get("first_token") is None:
            state = "reading"
        elif busy:
            state = "generating"
        elif not self.loaded():
            state = "unloaded"
        else:
            state = "idle"
        live = {"state": state, "queued": s.get("queued", 0),
                "prompt_tokens": s.get("prompt_tokens") if busy else None,
                "generated": s.get("generated") if busy else None,
                "elapsed_s": round(time.time() - s["started"], 1) if busy and s.get("started") else None,
                "tok_s": round(self._tok_s(), 1) if state == "generating" else None,
                "prefill_tok_s_mean": getattr(self.engine, "prefill_tok_s_mean", None) if busy else None}
        engine = {"model": self.model, "max_context": self.engine.max_context,
                  **dict(getattr(self.engine, "info", {}) or {})}
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else \
            {"now": {"gpus": []}, "history": {}, "static": {"gpus": []}}
        return {"engine": engine, "live": live, "requests": hist[::-1][:REQUESTS_SHOWN],
                "hardware": tel["now"], "hardware_static": tel["static"], "history": tel["history"]}

    def v1_status(self) -> dict:
        """GET /v1/status: what this server is and does, for a client that would rather ask than guess (a front-end
        that polls its OpenAI-compatible server's status, collabosm's for one): the model and its window, images,
        the APIs, what is running, and the last request's timings in llama.cpp's names.  /metrics has the rest."""
        with self.status_lock:
            s = dict(self.status)
            last_t, last_at = (dict(self.last_timings) if self.last_timings else None), self.last_request_at
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "static": {}}
        gpus, gpus_static = (tel.get("now") or {}).get("gpus") or [], (tel.get("static") or {}).get("gpus") or []
        hw, name = (gpus[0] if gpus else {}), (gpus_static[0].get("name") if gpus_static else None)
        ram = tel.get("now") or {}

        def scaled(v, unit, digits=0):
            return round(v / unit, digits) if isinstance(v, (int, float)) else None

        busy, ctx = bool(s.get("busy")), self.engine.max_context
        images = self.vision is not None
        return {
            "service": "basalt", "model": self.model,
            "loaded": self.loaded(), "auto_load": hasattr(self.engine, "restart"),
            "structured_output": {"formats": ["json_object", "json_schema"], "method": "prompt_and_validate",
                                  "constrained_decoding": False, "stream_buffered": True},
            "engine": (getattr(self.engine, "info", {}) or {}).get("version"),
            "started": int(self.started_at), "uptime_s": int(time.time() - self.started_at),
            "cache_max_tokens": ctx,
            "context": {"native": ctx, "max_positions": ctx},
            # one request at a time (more wait their turn), or "parallel": N batch slots
            "concurrency": {"serving": max(1, int(getattr(self.engine, "batch", 0) or 0)),
                            "requested": max(1, int(getattr(self.engine, "batch", 0) or 0))},
            "dialects": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
            "vision": {"enabled": images, "available": images, "error": None},
            "activity": {"requests": self.requests_done + int(busy), "in_flight": int(busy) + int(s.get("queued") or 0),
                         "last_request_at": int(last_at) if last_at else None},
            "last_timings": last_t,
            "vram": dict((getattr(self.engine, "info", {}) or {}).get("vram") or {},
                         elastic=bool((getattr(self.engine, "info", {}) or {}).get("vram_elastic"))),
            "machine": {
                "at": int(time.time()),
                "gpu": {"name": name, "used_mib": scaled(hw.get("mem_used"), 2 ** 20),
                        "total_mib": scaled(hw.get("mem_total"), 2 ** 20), "util_pct": hw.get("util"),
                        "temp_c": hw.get("temp"), "power_w": hw.get("power")} if name else None,
                "ram": {"used_gib": scaled(ram.get("ram_used"), 2 ** 30, 1),
                        "total_gib": scaled(ram.get("ram_total"), 2 ** 30, 1)} if ram.get("ram_total") else None}}

    def render_prompt(self, messages, tools, kwargs) -> str:
        """The template rendered.  With `effort_end`, a request with a non-default effort (low, medium or no
        thinking) is rendered as a default one up to the answer - the same prompt start, so the conversation cache
        keeps it - and its effort follows in a short system turn right before the answer (thinking off: the template's
        empty thinking block).  The engine (--tail-role-token) checkpoints in front of that turn."""
        effort = kwargs.get("reasoning_effort")
        off = kwargs.get("enable_thinking") is False
        if not self.effort_end or (not off and effort in (None, "", "xhigh", "high")):
            return self.template.render(messages, tools=tools, **kwargs)
        base = {k: v for k, v in kwargs.items() if k not in ("reasoning_effort", "enable_thinking")}
        history = self.template.render(messages, tools=tools, add_generation_prompt=False, **base)
        full = self.template.render(messages, tools=tools, add_generation_prompt=True, **kwargs)
        mine = self.template.render(messages, tools=tools, add_generation_prompt=False, **kwargs)
        if not full.startswith(mine):                  # a template this cannot take apart: rendered as asked
            return full
        tail = "" if off else EFFORT_TURN.format(EFFORT_TEXT.get(effort, EFFORT_TEXT["medium"]))
        return history + tail + full[len(mine):]

    def encode_prompt(self, messages, tools, kwargs) -> list[int]:
        """The request's prompt: the template rendered and tokenized.  A <think> / </think> written inside a
        message's text is encoded as the text it is, not as the model's reasoning markers (the template's own are).
        A control token's text (<|im_start|>, <|im_end|>, <|endoftext|>, ...) inside a message is text as well: only
        the control tokens the template writes are control tokens."""
        marked, marked_tools, changed = mark_think_literals(messages, tools, self.literals)
        prompt = self.render_prompt(marked, marked_tools, kwargs)
        if not changed:
            return self.tok.encode(prompt, parse_special=True)
        prompt, plain = unmark_think_literals(prompt, self.literals)
        return self.tok.encode(prompt, parse_special=True, plain=plain)

    def prepare(self, messages, tools, kwargs, max_new=None):
        """-> (ids, thinking, max_new). An unset or non-positive max_new (some clients send -1) means "unlimited":
        the rest of the context."""
        ids = self.encode_prompt(messages, tools, kwargs)
        self.embeddings.path = None
        images = images_of(messages)
        if images:
            if self.vision is None:
                raise ValueError("this server was started without image input (--vision or --vision-cpu), "
                                 "so it cannot read images")
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            start = self.tok.encode(VISION_START, parse_special=True)[0]
            # Encode only while the engine is idle: the engine and the image encoder (a separate process) must not
            # run on the GPU at the same time - an encode during a running request left that request stuck at
            # "reading the prompt" with CPU and GPU busy, for good (reproduced).  So encoding takes its turn in the
            # same FIFO as the requests.
            with self.fifo:
                encoded = [self.vision.encode(src) for src in images]
            # one <|image_pad|> per image -> one per image token.  Only the markers the template writes for an image
            # (right after <|vision_start|>) are images: the same text inside a message (an agent reading these docs,
            # say) is kept as plain text: it must not take an image's place, or the counts would not match.
            literal = self.tok.encode(IMAGE_PAD, parse_special=False)
            out, k = [], 0
            for j, t in enumerate(ids):
                if t == pad and j > 0 and ids[j - 1] == start:
                    # more pairs than images: text parts that cut both markers apart (whole ones are kept as text)
                    if k == len(encoded):
                        raise ValueError("the prompt and its images do not match")
                    out += [pad] * encoded[k][1]
                    k += 1
                elif t == pad:
                    out += literal
                else:
                    out.append(t)
            if k != len(encoded):
                raise ValueError("the prompt and its images do not match")
            ids = out
            combined = self.vision.dir / f"req-{uuid.uuid4().hex[:12]}.sve"
            with open(combined, "wb") as f:
                for path, _ in encoded:
                    f.write(path.read_bytes())
            self.embeddings.path = combined
        ctx = self.engine.max_context
        if ctx <= 0:
            if getattr(self.engine, "starting", False):   # (re)starting, not a prompt that is too long
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
            # a failed restart leaves max_context 0: check against the last known context, so the request reaches
            # run() - which starts the engine again - instead of failing with "exceeds the context (0)" forever
            ctx = getattr(self.engine, "known_ctx", 0)
            if ctx <= 0:
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
        room = ctx - CTX_SLACK - len(ids)
        if max_new is None or max_new <= 0 or (self.fit_max_tokens and room < 1):
            if room < 1:
                raise ValueError(f"prompt ({len(ids)} tokens) leaves no room to answer in the context "
                                 f"({ctx}); requests are never truncated")
            max_new = room
        elif max_new > room:
            if not self.fit_max_tokens:
                raise ValueError(f"prompt ({len(ids)} tokens) + max tokens ({max_new}) exceeds the context "
                                 f"({ctx}); requests are never truncated. Send a smaller "
                                 f"max_tokens (at most {max(0, room)} here), or add \"fit_max_tokens\": true to the "
                                 "model's <model>.json to shorten it to the room left")
            max_new = max(1, room)          # --fit-max-tokens: a shorter completion beats a 400
        return ids, kwargs.get("enable_thinking", True) is not False, max_new

    def _note(self, n, evs, st=None, rate=None):
        with self.status_lock:
            s = self.status if st is None else st
            s["generated"] = n
            if s.get("first_token") is None:
                s["first_token"] = time.time()
            (self.rate if rate is None else rate).append((time.time(), n))   # the live rate's window (RATE_WINDOW_S)
            for ev in evs:
                if ev.kind == "reasoning":
                    s["phase"] = "thinking"
                elif ev.kind == "content":
                    s["phase"] = "answering"
                elif ev.kind == "tool_start":
                    s["phase"], s["tool"] = f"writing a tool call: {ev.call.name}", ev.call.name
                elif ev.kind == "tool_call":
                    s["phase"] = "tool call complete"
                s["tail"] = ((s.get("tail") or "") + (ev.text or ""))[-600:]

    def _progress(self, last_print, every=1.0, st=None):
        """A progress line in the server window every `every` seconds while a request runs."""
        now = time.time()
        if now - last_print < every:
            return last_print
        with self.status_lock:
            s = dict(self.status if st is None else st)
        el = now - s.get("started", now)
        if s.get("first_token") is None:
            pr = getattr(self.engine, "progress", None)   # (position reached, prompt tokens): a reused prefix counts
            done = f"{pr[0]:,} of {pr[1]:,}" if pr and pr[1] else f"{s.get('prompt_tokens', 0):,}"   # as read
            print(f"[server] reading the prompt: {done} tokens, {el:.0f} s", flush=True)
        else:
            rate = s["generated"] / max(1e-6, now - s["first_token"])
            print(f"[server] {s['phase']}: {s['generated']} of max {s.get('max_tokens')} tokens, {rate:.1f} tok/s, "
                  f"{el:.0f} s", flush=True)
        return now

    def run(self, ids, thinking, tools, max_new, sampling, cancel) -> Iterator[tuple[str, object]]:
        """Yields ("event", Event) as text arrives, then ("done", {"finish": .., "completion_tokens": ..})."""
        budget = self.reasoning_budget(sampling) if thinking else None   # opt-in, off by default
        if self.sampling_defaults:     # the request's own fields win (explicit 0 stays greedy)
            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}
            sampling = {**self.sampling_defaults, **req_values}
        parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True)
        detok, n, finish = Detokenizer(self.tok), 0, "length"
        run_tok, run_len, repeated = None, 0, False     # the current run of one repeated token
        thinking_n = 0                                  # tokens written while thinking (Responses' reasoning_tokens)
        stop_list = stop_strings(sampling)
        stops = StopMatcher(stop_list) if stop_list else None

        def cut(evs):
            """The events with the answer's text cut at a stop string (OpenAI stop, Anthropic stop_sequences)."""
            if stops is None:
                return evs
            out = []
            for ev in evs:
                if stops.hit is not None:
                    break                               # nothing after the stop string is sent
                if ev.kind == "content":
                    text = stops.push(ev.text or "")
                    if text:
                        out.append(Event("content", text))
                    continue
                held = stops.flush()                    # a stop string does not run across a tool call or thinking
                if held:
                    out.append(Event("content", held))
                out.append(ev)
            return out
        timings, before = None, None                    # this request's timings; the engine's `last` before it
        raw_ids = []                                    # every generated id (--debug: dump raw model text)
        emb = getattr(self.embeddings, "path", None)
        # Identity token: only a DONE line replaces engine.last, so a request that died, errored or was
        # disconnected must not have the PREVIOUS request's decode figures recorded as its own.
        engine_last0 = getattr(self.engine, "last", None)
        path = getattr(self.request_ctx, "path", None)   # the API path serving this request (GET /metrics)
        # with "parallel" (the engine's batch slots) requests run at once: each keeps its own status and rate
        # window (self.status says busy while any runs); one at a time they are self.status / self.rate, as before
        par = bool(getattr(self.engine, "batch", 0))
        st = {} if par else self.status
        rate = collections.deque(maxlen=32) if par else self.rate
        with self.status_lock:
            self.status["queued"] += 1
        try:
            # --batch: the engine runs several requests at once (ProcessEngine.generate_batched orders them)
            with (contextlib.nullcontext() if getattr(self.engine, "batch", 0) else self.fifo):
                try:
                    with self.status_lock:
                        self.status["queued"] -= 1
                    # it died in an earlier request (or was unloaded) - start it again instead of failing
                    # (parallel requests do not hold the fifo: one of them starts it, the others wait for that)
                    with (self.fifo if par else contextlib.nullcontext()):
                        self.ensure_loaded()
                    with self.status_lock:
                        st.update(busy=True, phase="reading the prompt", prompt_tokens=len(ids),
                                  generated=0, started=time.time(), first_token=None, tool=None, tail="",
                                  max_tokens=max_new)
                        if par:
                            self.live_reqs[id(st)] = (st, rate)
                            self.status.update(st)
                        self.last_request_at = time.time()
                        rate.clear()                    # the previous request's samples must not leak into this one
                    before = getattr(self.engine, "last", None)
                    last_print = time.time()
                    prompt, thought = ids, 0            # thought: the reasoning tokens so far (the budget's count)
                    closed_early = False                # the thinking was closed once after the turn ended in it
                    while True:
                        gen = self.engine.generate(prompt, max_new - n, sampling, cancel, embeddings=emb) if emb \
                            else self.engine.generate(prompt, max_new - n, sampling, cancel)
                        seg, wrap, leaving = [], False, False   # this pass's tokens; the budget is reached; closed
                        ended_thinking = False          # the turn ended inside the thinking: close it and answer
                        try:
                            for t in gen:
                                if t is None:               # heartbeat while the engine is quiet
                                    last_print = self._progress(last_print, st=st)
                                    yield "ping", None
                                    continue
                                n += 1
                                if t in self.stop_ids:
                                    if (parser.state == "reasoning" and not closed_early and not parser.buf
                                            and not detok.pending()):
                                        n -= 1              # the end token is dropped, not part of the reply
                                        ended_thinking = True
                                        break
                                    finish = "stop"
                                    raw_ids.append(t)
                                    break
                                raw_ids.append(t)
                                seg.append(t)
                                thinking_n += parser.state == "reasoning"
                                run_len = run_len + 1 if t == run_tok else 1
                                run_tok = t
                                if self.repeat_stop_tokens and run_len >= self.repeat_stop_tokens:
                                    repeated = True     # a degenerate output, not an answer: end it here
                                    break
                                evs = cut(parser.feed(detok.push(t)))
                                self._note(n, evs, st, rate)
                                last_print = self._progress(last_print, st=st)
                                for ev in evs:
                                    yield "event", ev
                                if stops is not None and stops.hit is not None:
                                    finish = "stop"         # gen.close() below STOPs the engine, as for a stop token
                                    break
                                if budget and parser.state == "reasoning":
                                    thought += 1
                                    # at a clean point: no tag held back, no character split across tokens
                                    if thought >= budget and not parser.buf and not detok.pending():
                                        wrap = True
                                        break
                        except EngineDied as e:
                            finish = "error"
                            self._say_died(e)
                            raise
                        except ValueError as e:             # the engine's ERR line (it may have ended after it)
                            finish = "error"
                            print(f"[server] error: engine reported {e}", flush=True)
                            raise
                        except GeneratorExit:               # the client went away: an engine that never acknowledges
                            leaving = True                  # the STOP below is ended, but no error replaces this
                            raise
                        finally:
                            try:
                                gen.close()             # STOP+drain to THIS request's DONE while still holding the
                                #                         fifo, so a stop-token break can't leave the shared engine
                                #                         queue mid-drain for the next request to read as its own DONE
                            except EngineSilent as e:   # the STOP was never acknowledged: the engine is ended
                                finish = "error"
                                self._say_died(e)
                                if not leaving and not cancel.is_set():
                                    raise
                        if not (wrap or ended_thinking) or cancel.is_set():
                            break
                        # the thinking reached reasoning_budget_tokens, or the model ended its turn inside it.  Close
                        # it the way the model would (for the budget, a short wrap-up first) and let it answer: the
                        # next pass's prompt is this one plus what was generated plus the close, so the engine
                        # continues from the prefix it already holds.
                        if ended_thinking:
                            closed_early = True
                            extra = self.tok.encode(REASONING_CLOSE, parse_special=True)
                            if max_new - n - len(extra) < 1:
                                finish = "stop"
                                break
                            print("[server] the model ended while thinking, closing it for an answer", flush=True)
                        else:
                            budget = None
                            extra = self.tok.encode(REASONING_WRAP_UP, parse_special=True)
                            if max_new - n - len(extra) < 1:
                                break                   # no room left to answer: "length", as without a budget
                            print(f"[server] thinking budget reached ({thought} tokens), wrapping up",
                                  flush=True)
                        for t in extra:
                            n += 1
                            raw_ids.append(t)
                            thinking_n += parser.state == "reasoning"
                            evs = cut(parser.feed(detok.push(t)))
                            self._note(n, evs, st, rate)
                            for ev in evs:
                                yield "event", ev
                        if stops is not None and stops.hit is not None:
                            finish = "stop"
                            break
                        prompt = prompt + seg + extra
                    if cancel.is_set():
                        finish = "cancel"
                    elif repeated:
                        print(f"[server] warning: the reply repeated one token ({self.tok.decode([run_tok])!r}) "
                              f"{run_len} times, stopped as length (repeat_stop_tokens, 0 = off)", flush=True)
                except GeneratorExit:                   # the client disconnected mid-stream
                    finish = "disconnect"
                    raise
                finally:
                    # settle this request's status and history while still holding the fifo: once
                    # it is released the next request sets its own status, which this must not record or clear
                    with self.status_lock:
                        if st.get("busy"):
                            # only this request's DONE counts: same object means no DONE arrived (death, error,
                            # disconnect)
                            last = dict(getattr(self.engine, "last", {}) or {}) \
                                if getattr(self.engine, "last", None) is not engine_last0 else {}
                            started = st.get("started", time.time())
                            hit_rate = round(last["hits"] / last["lookups"], 3) if last.get("lookups") else None
                            # the share the GPU read over PCIe (--pcie-frac) or another GPU computed - not in
                            # the hit rate's lookups, which is the VRAM share of the rest (None: an older engine)
                            routed = (last.get("lookups") or 0) + (last.get("offloaded") or 0)
                            pcie_share = round(last["offloaded"] / routed, 3) \
                                if last.get("offloaded") is not None and routed else None
                            seen = prompt_tokens_seen(len(ids), last)   # < len(ids) when cancelled mid-read
                            self.requests_done += 1
                            self.history.append({
                                "id": f"r_{self.requests_done:04d}", "path": path,
                                "time": started, "duration_s": round(time.time() - started, 1), "finish": finish,
                                "prompt_tokens": seen, "reused": last.get("reused"), "output_tokens": n,
                                "prompt_ms": last.get("prompt_ms"),
                                "decode_tok_s": round(last["generated"] / (last["decode_ms"] / 1000), 1)
                                if n and last.get("generated") and last.get("decode_ms") else None,
                                "hit_rate": hit_rate, "pcie_share": pcie_share,
                                # the speculative drafts from the DONE line (None: the engine did not say)
                                "drafts_offered": last.get("drafts_offered"),
                                "drafts_accepted": last.get("drafts_accepted")})
                            fresh = getattr(self.engine, "last", None)
                            if fresh is not None and fresh is not before:      # the engine's clock for THIS request
                                timings = request_timings(seen, n, last)
                                self.last_timings = dict(timings, at=int(time.time())) if timings else None
                            self.last_request_at = time.time()
                            now = time.time()
                            el = now - st.get("started", now)
                            ft = st.get("first_token")
                            rate = n / max(1e-6, now - ft) if ft else 0.0
                            hit_msg = f", expert cache {hit_rate*100:.1f}% hit" if hit_rate is not None else ""
                            if hit_msg and pcie_share:
                                hit_msg += f" (+{pcie_share*100:.1f}% of the routed experts over PCIe)"
                            print(f"[server] done: {n} tokens in {el:.0f} s ({rate:.1f} tok/s) "
                                  f"({finish}, cancel={cancel.is_set()}){hit_msg}", flush=True)
                            if finish == "length" and parser.state == "reasoning":   # 0
                                print("[server] warning: max tokens reached while thinking, no answer "
                                      "(set reasoning_budget_tokens)", flush=True)
                            if DEBUG and raw_ids:
                                print(f"[server] raw: {self.tok.decode(raw_ids)!r}", flush=True)
                        st["busy"] = False
                        st.pop("tail", None)                     # the answer's end is not kept once it is done
                        st.pop("tool", None)
                        if par:
                            self.live_reqs.pop(id(st), None)
                            if self.live_reqs:              # the newest request still running
                                self.status.update(list(self.live_reqs.values())[-1][0])
                            else:
                                self.status.update(busy=False)
                                self.status.pop("tail", None)
                                self.status.pop("tool", None)
        finally:
            if emb:
                Path(emb).unlink(missing_ok=True)
        for ev in cut(parser.finish()):
            yield "event", ev
        if stops is not None and stops.hit is None and stops.held:
            yield "event", Event("content", stops.flush())     # the held tail was not a stop string after all
        done = {"finish": finish, "completion_tokens": n, "reused": (timings or {}).get("cache_n", 0),
                "timings": timings, "reasoning_tokens": thinking_n}
        if stops is not None and stops.hit is not None:
            done["stop_sequence"] = stops.hit
        yield "done", done


def prompt_tokens_seen(prompt_tokens: int, last: dict) -> int:
    """The prompt tokens a request got through - all of them, unless the engine's DONE line says a cancel stopped
    its prompt read part-way (then the reused ones plus those read).  /metrics' history counts these, so a
    cancelled read is neither recorded as the whole prompt nor given a rate from tokens it never read.  An engine
    that does not say (no `prompt_read`): the whole prompt, as before."""
    read = last.get("prompt_read")
    if read is None or last.get("finish") != "cancel":
        return prompt_tokens
    return min(prompt_tokens, int(last.get("reused") or 0) + int(read))


def request_timings(prompt_tokens: int, generated: int, last: dict) -> dict | None:
    """One request's `timings` in llama.cpp's names (what its clients show as speed), from the engine's own clock
    (ProcessEngine.last): prompt_n is what was read, cache_n what the conversation cache already held.  None when the
    engine keeps no clock (MockEngine)."""
    if last.get("prompt_ms") is None:
        return None
    cache_n = int(last.get("reused") or 0)
    prompt_n, prompt_ms, decode_ms = max(0, prompt_tokens - cache_n), float(last["prompt_ms"]), float(last.get("decode_ms") or 0)
    decoded = int(last.get("generated") or generated)          # the engine's count gives its rate, as /metrics does
    return {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 1),
            "prompt_per_token_ms": round(prompt_ms / prompt_n, 3) if prompt_n else None,
            "prompt_per_second": round(prompt_n / (prompt_ms / 1000), 1) if prompt_n and prompt_ms > 0 else None,
            "predicted_n": generated, "predicted_ms": round(decode_ms, 1),
            "predicted_per_token_ms": round(decode_ms / decoded, 3) if decoded else None,
            "predicted_per_second": round(decoded / (decode_ms / 1000), 1) if decoded and decode_ms > 0 else None,
            # the speculative drafts, as llama.cpp names them: only when the engine reported them
            **({"draft_n": int(last["drafts_offered"]), "draft_n_accepted": int(last["drafts_accepted"])}
               if last.get("drafts_offered") is not None else {})}


def _debug_req(api, req, messages, tools, max_new, thinking, prompt_tokens):
    """One compact line per request while diagnosing blank/empty turns. Enabled by --debug."""
    if not DEBUG:
        return
    last = messages[-1] if messages else {}
    body = last.get("content")
    if isinstance(body, list):
        body = " ".join(p.get("text", "") for p in body if isinstance(p, dict))
    preview = (str(body or "")[:80]).replace("\n", " ")
    print(f"[server] req {api}: msgs={len(messages)} tools={len(tools or [])} "
          f"max_tokens_raw={req.get('max_tokens')!r}/{req.get('max_completion_tokens')!r} "
          f"max_new={max_new} thinking={thinking} stream={bool(req.get('stream'))} "
          f"prompt_tokens={prompt_tokens} last={last.get('role')!r}:{preview!r}", flush=True)
