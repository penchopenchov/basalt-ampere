"""OpenAI, Anthropic and Responses endpoints over an engine that maps token ids to tokens.

    basalt serve --model FILE.basalt [--alias NAME] [--host H] [--port P] [--gpu 0,1] [--api-key KEY] ...
    python -m serve.server --model FILE.basalt --engine-bin bin/basalt ...
    python -m serve.server --config config.json
    python -m serve.server --mock-engine --port 8095

The engine boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel)` -> token ids. ProcessEngine keeps
one `basalt --serve` process resident; MockEngine is a scripted stand-in for tests. Prompts that do not fit the
context are rejected with a 400, never truncated.
"""
from __future__ import annotations

import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # also runnable as a script
from serve import ROOT, service  # noqa: E402
from serve.config import (build_parser, check_engine_flags, engine_flags_of, engine_line, engine_settings,  # noqa: E402
                          parse_settings, sampling_defaults_from_config)
from serve.engine import process as engine_process  # noqa: E402
from serve.engine.errors import EngineStartError  # noqa: E402
from serve.engine.mock import MockEngine  # noqa: E402
from serve.engine.process import ProcessEngine  # noqa: E402
from serve.engine.vision import (Vision, child_env, engine_args, gpu_list, gpus_in_play, vision_env,  # noqa: E402
                                 vision_line)
from serve.frontend import ChatTemplate  # noqa: E402
from serve.handler import Server, serve  # noqa: E402
from serve.security import LOOPBACK_NAMES, allowed_hosts_of, exposed_without_key, lan_addresses, origins_of  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.logbuf import LogBuffer, TeeStream, tail_file  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402


def fail(msg: str) -> None:
    """Print a startup error and exit with status 2."""
    print(f"[server] error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def warn_tight_ram(arena_mib) -> None:
    """Warn at start when less than ~6 GB of RAM is left beside the expert arena (INFO arena_mib)."""
    if not isinstance(arena_mib, int) or arena_mib <= 0:
        return
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - psutil is optional here
        return
    left = total / 2**30 - arena_mib / 1024
    if left < 6:
        print(f"[server] warning: low RAM, {left:.1f} of {total / 2**30:.0f} GB left after the experts", flush=True)


def main() -> int:
    ap = build_parser()
    try:
        a, used_argv, file_engine, cli_engine = parse_settings(ap, sys.argv[1:])
    except ValueError as e:
        fail(f"--config: {e}")
    service.DEBUG, engine_process.REQUEST_LINES = a.debug, a.request_lines
    logs = LogBuffer()                                  # this process's and the engine's output
    sys.stdout, sys.stderr = TeeStream(sys.stdout, logs, "srv"), TeeStream(sys.stderr, logs, "srv")
    cfg = {}                                            # what the model file implies
    try:
        if a.gpu is not None:
            cfg["gpu"] = a.gpu
            gpu_list(cfg)
    except ValueError:
        fail(f"--gpu {a.gpu}: expected numbers like 0 or 0,1")
    if not a.mock_engine and not (a.model and a.engine_bin):
        fail("the server needs --model and the engine binary: run it as basalt serve --config FILE (or --mock-engine)")
    if (file_engine or cli_engine) and not a.model:
        fail("engine flags need --model")
    if a.model:
        if not a.engine_bin:
            fail("--model needs --engine-bin BINARY")
        from serve.engine import modelfile
        listed = gpu_list(cfg)
        try:
            known = engine_flags_of(a.engine_bin)
            check_engine_flags(file_engine, known, f"--config {a.config}")
            check_engine_flags(cli_engine, known, "command line")
        except ValueError as e:
            fail(str(e))
        flags, sources = engine_settings(file_engine, cli_engine)
        try:
            cfg.update({k: v for k, v in modelfile.model_config(
                a.model, a.engine_bin, gpus=len(listed) if listed else None, vision=("cpu" if a.vision_cpu else True) if not a.mock_engine and (a.vision or a.vision_cpu) else False,
                overrides={f: v for f, v in flags.items() if v is not False}).items()
                if k not in cfg})
        except (OSError, ValueError, KeyError) as e:
            fail(f"{a.model}: {e}")
        off = {f for f, v in flags.items() if v is False}
        if off and cfg.get("args"):
            cfg["args"] = [x for x in cfg["args"] if x not in off]
        if not a.mock_engine:
            print(engine_line(flags, sources, bool(a.config)), flush=True)
    if a.alias:
        cfg["model_name"] = a.alias
    try:                                                # fail fast if the port is taken
        Server((a.host, a.port), BaseHTTPRequestHandler).server_close()
    except OSError:
        fail(f"port {a.port} is in use (--port N)")
    if cfg.get("tokenizer"):
        a.tokenizer = cfg["tokenizer"]
    tok = ByteTokenizer()
    tpath = Path(a.tokenizer)
    if not a.mock_engine and not (tpath / "vocab.json").exists():
        fail(f"tokenizer missing: {tpath / 'vocab.json'}")
    if (tpath / "vocab.json").exists():
        import tokenizer as ST
        vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tpath / "token_type.json").read_text())
        tok = ST.Tokenizer(tokens, merges, types)
    if not a.mock_engine:
        vision = None
        env = child_env(cfg)
        sampling_defaults = sampling_defaults_from_config(cfg)
        if sampling_defaults:
            pretty = ", ".join(f"{k}={v}" for k, v in sampling_defaults.items())
            print(f"[server] sampling defaults: {pretty}", flush=True)
        lazy = a.lazy
        if lazy and cfg.get("vision"):
            print("[server] warning: lazy loading is text-only, vision off", flush=True)
            del cfg["vision"]
        if cfg.get("vision"):
            vision = Vision(cfg["vision"], log=open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None,
                            env=vision_env(cfg, env))
        print(vision_line(cfg.get("vision"), cfg, vision.ready_s if vision else 0.0), flush=True)
        if lazy:
            print("[server] model unloaded, the first request loads it", flush=True)
        if cfg.get("log"):                              # follow the engine's log file
            threading.Thread(target=tail_file, args=(logs, cfg["log"], "eng"), daemon=True).start()
        else:
            ProcessEngine.stderr_sink = lambda line: logs.add("eng", line)
        try:
            engine = ProcessEngine(cfg["engine_bin"], engine_args(cfg), log=cfg.get("log"), env=env, lazy=lazy)
        except EngineStartError as e:
            print(f"[server] error: {e}", file=sys.stderr, flush=True)
            if e.log:
                print(f"[server] engine log: {e.log}", file=sys.stderr, flush=True)
            return 1
        warn_tight_ram(engine.info.get("arena_mib"))
    else:
        engine, vision, sampling_defaults = MockEngine(tok, a.script or [
            "Thinking about it.</think>\n\nHello from the mock engine."]), None, {}
    # the model's exported template, else the bundled one
    tpl = tpath / "chat_template.jinja"
    svc = Service(engine, tok, ChatTemplate(tpl if tpl.exists() else ROOT / "serve/chat_template.jinja"),
                  model_name=cfg.get("model_name", "qwen3.8-flash-next"), vision=vision,
                  sampling_defaults=sampling_defaults, fit_max_tokens=a.fit_max_tokens)
    svc.logs = logs
    if any(x == "--api-key" and i + 1 < len(used_argv) and not used_argv[i + 1].strip() or x.strip() == "--api-key="
           for i, x in enumerate(used_argv)):
        # an empty key would silently disable authentication
        print("[server] error: --api-key is empty (give a key, or leave it out)", file=sys.stderr)
        return 2
    svc.api_key = a.api_key
    try:
        svc.cors_origins = origins_of(a.cors_origin, "--cors-origin", wildcard=True)
        svc.allowed_hosts = allowed_hosts_of(a.allowed_host)
    except (ValueError, SystemExit) as e:
        fail(str(e).removeprefix("[server] error: "))
    if svc.allowed_hosts:
        print("[server] warning: Host check off, any name is accepted (--allowed-host *)" if "*" in svc.allowed_hosts
              else f"[server] extra host names: {', '.join(svc.allowed_hosts)}", flush=True)
    if svc.cors_origins:
        print(f"[server] CORS on /v1/* for {', '.join(svc.cors_origins)}"
              + ("" if svc.api_key or "*" not in svc.cors_origins else
                 "; warning: any web page can use the model (--api-key KEY)"), flush=True)
    svc.idle_unload_s = a.idle_unload or 0
    svc.min_free_vram_mib = a.min_free_vram_mib or 0
    svc.gpu_indices = gpus_in_play(cfg, engine_args(cfg) if not a.mock_engine else [])   # the cards /metrics reads
    svc.gpu_index = svc.gpu_indices[0]
    httpd = serve(svc, host=a.host, port=a.port)
    svc.start_idle_unload()
    here = "127.0.0.1" if a.host in ("0.0.0.0", "", "::") else a.host
    print(f"[server] ready: http://{here}:{a.port}/v1 (OpenAI /v1/chat/completions, Anthropic /v1/messages), "
          f"context {engine.max_context} tokens{', images on' if vision else ''}"
          f"{', API key required' if svc.api_key else ''}", flush=True)
    print(f"[server] stats: http://{here}:{a.port}/ (Ctrl+C stops)", flush=True)
    if a.host not in LOOPBACK_NAMES:
        # reachable from other devices: print the addresses
        ips = lan_addresses()
        for ip in ips:
            print(f"[server] network: http://{ip}:{a.port}/v1", flush=True)
        if not ips:
            print(f"[server] network: port {a.port}", flush=True)
        if exposed_without_key(a.host, svc.api_key):
            print("[server] warning: no API key, open to the network (--api-key KEY)", flush=True)
    if a.open:
        import webbrowser
        webbrowser.open(f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}/")
    # SIGTERM takes Ctrl+C's path (QUIT to the engine) so a container stop does not orphan it
    def on_sigterm(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, on_sigterm)
    except (ValueError, OSError, AttributeError):         # not the main thread
        pass
    try:
        while True:
            time.sleep(1)                               # a timed wait keeps Ctrl+C deliverable
    except KeyboardInterrupt:
        print("[server] stopping (Ctrl+C again to force)", flush=True)
        closers = [httpd.shutdown, getattr(engine, "close", None), vision.close if vision else None]
        for close in filter(None, closers):
            try:
                close()
            except KeyboardInterrupt:                   # second Ctrl+C: do not wait for the engine
                if getattr(engine, "proc", None):
                    engine.proc.kill()
        print("[server] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
