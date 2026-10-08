"""Server settings: the command-line parser, the --config file, engine flag checks and the startup lines about them."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from serve import ROOT
from serve.security import exposed_without_key


ENGINE_DEFAULT_CONTEXT = 4096     # the engine's context without --max-context
# server options that are really engine flags: their home in the config file is "engine"
ENGINE_PASSTHROUGH = ("spec", "spec-min-p", "mtp-max-t", "peer-exclusive", "max-context")
CONFIG_PATHS = ("model", "tokenizer")       # resolved against the config file's directory


def engine_flags_of(engine_bin: str) -> set[str] | None:
    """The long flags the engine binary accepts (the option strings stored in it); None when it cannot be read."""
    try:
        with open(engine_bin, "rb") as f:
            data = f.read()
    except OSError:
        return None
    return {m.decode() for m in re.findall(rb"(?<![\x21-\x7e])--[a-z0-9][a-z0-9-]*(?![\x21-\x7e])", data)}


def server_options(ap: argparse.ArgumentParser) -> dict:
    """{name: action} of the server's own options (names without the dashes), apart from --config and --help."""
    return {o[2:]: act for act in ap._actions for o in act.option_strings
            if o.startswith("--") and o[2:] not in ("help", "config")}


def split_engine_extras(extras: list[str]) -> dict:
    """The engine flags among the command-line words the server does not know: `--flag VALUE`, `--flag=VALUE` or a
    bare `--flag` (a switch).  A word after a flag that does not start with `--` is its value.  ValueError for a
    stray word."""
    out, i = {}, 0
    while i < len(extras):
        word = extras[i]
        if not word.startswith("--") or len(word) < 3:
            raise ValueError(f"unexpected {word!r}")
        flag, eq, value = word.partition("=")
        if eq:
            out[flag] = value
        elif i + 1 < len(extras) and not extras[i + 1].startswith("--"):
            out[flag] = extras[i + 1]
            i += 1
        else:
            out[flag] = True
        i += 1
    return out


def load_config_file(path: str, ap: argparse.ArgumentParser) -> tuple[list[str], dict]:
    """(argv, engine) from a --config JSON file: {"server": {...}, "engine": {...}}.  `server` holds the server's
    options (name with or without the dashes -> a string or number, true/false for a switch, a list for the
    repeatable ones); it comes back as the command-line words they stand for, so every command-line check applies
    to it.  `engine` holds engine flags ({"--flag": value}, true for a switch, false to leave it off).  model, engine-bin
    and tokenizer paths are relative to the file.  ValueError for anything else."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except OSError as e:
        raise ValueError(f"cannot read: {e.strerror or e}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from None
    if not isinstance(raw, dict) or set(raw) - {"server", "engine"}:
        raise ValueError('expected {"server": {...}, "engine": {...}}'
                         + (f", not the key {sorted(set(raw) - {'server', 'engine'})[0]!r}" if isinstance(raw, dict) else ""))
    sections = {k: raw.get(k, {}) for k in ("server", "engine")}
    for k, v in sections.items():
        if not isinstance(v, dict):
            raise ValueError(f'"{k}" must be an object')
    here = Path(path).resolve().parent
    known = server_options(ap)
    argv: list[str] = []
    for key in (*sections["server"], *sections["engine"]):
        if re.sub(r"[^a-z]", "", str(key).lower()).startswith("apikey"):
            raise ValueError("api-key can't be set in a config file; pass --api-key on the command line")
    for key, value in sections["server"].items():
        name = str(key).strip().lower().replace("_", "-").removeprefix("--")
        act = known.get(name)
        if name == "engine-bin":
            raise ValueError("server.engine-bin: engine-bin is set by basalt serve")
        if name in ENGINE_PASSTHROUGH:
            raise ValueError(f'server.{key}: an engine flag, set it under "engine"')
        if act is None:
            raise ValueError(f"server.{key}: not a server option")
        flag = "--" + name
        if act.nargs == 0:                                 # a switch
            if not isinstance(value, bool):
                raise ValueError(f"server.{key}: true or false")
            argv += [flag] if value else []
            continue
        items = value if isinstance(value, list) else [value]
        if isinstance(value, list) and not isinstance(act, argparse._AppendAction):
            raise ValueError(f"server.{key}: one value, not a list")
        for item in items:
            if isinstance(item, bool) or not isinstance(item, (str, int, float)):
                raise ValueError(f"server.{key}: expected a string or number, not {item!r}")
            item = str(item)
            if name in CONFIG_PATHS and item:
                item = str((here / os.path.expanduser(item)))
            argv += [flag, item]
    engine = {}
    for key, value in sections["engine"].items():
        flag = "--" + str(key).strip().lower().replace("_", "-").removeprefix("--")
        if flag[2:] in known and flag[2:] not in ENGINE_PASSTHROUGH:
            raise ValueError(f'engine.{key}: a server option, set it under "server"')
        if isinstance(value, bool) or (isinstance(value, (str, int, float)) and str(value).strip() != ""):
            engine[flag] = value
        else:
            raise ValueError(f"engine.{key}: expected a string or number, or true/false for a switch, not {value!r}")
    return argv, engine


def parse_settings(ap: argparse.ArgumentParser, cli: list[str]):
    """(args, argv, file_engine, cli_engine): the parsed server options and the words they came from (the config
    file's, then the command line's); the engine flags the config file and the command line set.  Command line >
    config file > defaults.  SystemExit(2) from argparse for a bad value; ValueError for a bad config or extra word."""
    a_cli, extras = ap.parse_known_args(cli)
    cli_engine = split_engine_extras(extras)
    file_argv, file_engine = [], {}
    if a_cli.config:
        file_argv, file_engine = load_config_file(a_cli.config, ap)
    a = ap.parse_known_args(file_argv + cli)[0]
    for act in ap._actions:                                # a repeatable option on the command line replaces the file's
        if isinstance(act, argparse._AppendAction) and getattr(a_cli, act.dest) != act.default:
            setattr(a, act.dest, getattr(a_cli, act.dest))
    for flag, value in (("--spec", a.spec), ("--spec-min-p", a.spec_min_p), ("--mtp-max-t", a.mtp_max_t),
                        ("--max-context", a.max_context), ("--peer-exclusive", a.peer_exclusive or None)):
        if value is not None:
            cli_engine[flag] = value
    host_on_cli = any(w == "--host" or w.startswith("--host=") for w in cli)
    if "--host" in file_argv and not host_on_cli and exposed_without_key(a.host, a.api_key):
        raise ValueError(f"host {a.host} is open to the network and comes from the config file: "
                         "pass --api-key on the command line")
    return a, file_argv + cli, file_engine, cli_engine


def check_engine_flags(flags: dict, known: set[str] | None, where: str) -> None:
    """ValueError naming the first flag the engine does not accept (unchecked when the binary cannot be read)."""
    for flag in flags:
        if known is not None and flag not in known:
            raise ValueError(f"{where}: {flag} is not an engine flag")


def engine_settings(file_flags: dict, cli_flags: dict) -> tuple[dict, dict]:
    """(flags, sources): the config file's flags overridden by the command line's, and where each came from."""
    flags, sources = {}, {}
    for src, items in (("config", file_flags), ("cli", cli_flags)):
        for flag, value in items.items():
            flags[flag], sources[flag] = value, src
    return flags, sources


def engine_line(flags: dict, sources: dict, config: bool = True) -> str:
    """The startup line listing the engine settings that came from --config or the command line.  Without a
    config file it warns that the engine's untuned built-in defaults apply."""
    shown = ", ".join(f"no {f} ({sources[f]})" if v is False else f"{f}{'' if v is True else f' {v}'} ({sources[f]})"
                      for f, v in flags.items())
    if config:
        return f"[server] engine: {shown}" if shown else "[server] engine: config sets nothing, built-in defaults"
    spec = flags.get("--spec", 0)
    state = f"spec {'off' if not int(spec) else spec}, {flags.get('--max-context', ENGINE_DEFAULT_CONTEXT)} context"
    return f"[server] engine: built-in defaults (no --config): {state}" + (f"; {shown}" if shown else "")


def sampling_defaults_from_config(cfg: dict) -> dict:
    """The model file's optional `sampling` block: defaults for the sampling fields a request leaves out, so
    a plain client gets the model's sampling instead of greedy.  Supported: temperature, top_p, top_k, min_p,
    presence_penalty, repetition_penalty, frequency_penalty, penalty_last_n, seed.  The request's own fields
    always win - an explicit temperature=0 still means greedy, a field set to null falls back to the default.
    A bad value refuses to start the server (a typo'd value should not quietly change sampling); unknown keys
    are named at startup and ignored."""
    out = {}
    for key, value in (cfg.get("sampling") or {}).items():
        if value is None:
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "temperature":
            if not number or value < 0:
                raise SystemExit(f"[server] error: sampling.temperature={value!r}: expected a number >= 0 (0 = greedy)")
            out[key] = float(value)
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise SystemExit(f"[server] error: sampling.top_p={value!r}: expected 0 < top_p <= 1")
            out[key] = float(value)
        elif key == "min_p":
            if not number or not 0 <= value <= 1:
                raise SystemExit(f"[server] error: sampling.min_p={value!r}: expected 0 <= min_p <= 1")
            out[key] = float(value)
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise SystemExit(f"[server] error: sampling.top_k={value!r}: the sampled path takes an integer 1..64")
            out[key] = int(value)
        elif key == "presence_penalty":
            if not number or value < 0:
                raise SystemExit(f"[server] error: sampling.presence_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "frequency_penalty":
            if not number or value < 0:
                raise SystemExit(f"[server] error: sampling.frequency_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "repetition_penalty":
            if not number or value <= 0:
                raise SystemExit(f"[server] error: sampling.repetition_penalty={value!r}: expected a number > 0 (1 = off)")
            out[key] = float(value)
        elif key == "penalty_last_n":
            if not number or value != int(value) or value < 0:
                raise SystemExit(f"[server] error: sampling.penalty_last_n={value!r}: expected a non-negative integer")
            out[key] = int(value)
        elif key == "seed":
            if not number or value != int(value) or value <= 0:
                raise SystemExit(f"[server] error: sampling.seed={value!r}: expected a positive integer")
            out[key] = int(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise SystemExit(f"[server] error: sampling.experimental_speed_projection={value!r}: expected true or "
                                 "false (the default for requests that leave it out, when the engine has the vector)")
            out[key] = value
        else:
            print(f"[server] warning: sampling.{key} is not a known key, ignored", flush=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--mock-engine", action="store_true",
                    help="run a scripted engine instead of the model, for clients and tests (no --model needed)")
    ap.add_argument("--model", help="the .basalt model file: the engine, tokenizer, sampling and vision come from it")
    ap.add_argument("--engine-bin", help=argparse.SUPPRESS)      # set by `basalt serve`; given by hand for development
    ap.add_argument("--alias", help="the model name the API reports (default: the name stored in the model file)")
    ap.add_argument("--spec", type=int, help="tokens per verify window, the drafter's cap (default: the model file's; "
                                             "0 = no speculation, the drafter is not loaded)")
    ap.add_argument("--spec-min-p", type=float, help="stop drafting below this draft probability (default: the "
                                                     "model file's; 0 = always draft to the cap)")
    ap.add_argument("--mtp-max-t", type=int, help="the drafter's own window cap (default 0 = --spec)")
    ap.add_argument("--max-context", type=int, help="the longest context the engine allows, in tokens (default: the "
                                                    "engine's)")
    ap.add_argument("--config", metavar="FILE",
                    help="a JSON object of engine flags, e.g. {\"max-context\": 32768, \"spec\": 7, "
                         "\"resident-experts\": true}; the server's own flags win over it")
    ap.add_argument("--vision", action="store_true",
                    help="image input, encoded on the GPU (off by default: the encoder holds VRAM; with two cards "
                         "listed in --gpu it uses the second)")
    ap.add_argument("--vision-cpu", action="store_true",
                    help="image input, encoded on the CPU (no VRAM, slower; wins over --vision)")
    ap.add_argument("--peer-exclusive", action="store_true",
                    help="the peer card's experts live only there, not also in RAM: less RAM, slower prefill")
    ap.add_argument("--host", default="127.0.0.1",
                    help="the address to listen on: 127.0.0.1 = this PC only (the default), 0.0.0.0 = also other "
                         "devices on your network (give --api-key)")
    ap.add_argument("--script", action="append",
                    help="the mock engine's answer (default: a short greeting); given more than once, requests get "
                         "them in turn and the last one repeats")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--gpu", help="the GPUs to use, as nvidia-smi numbers them: the first runs the model, a second is "
                                  "the peer expert tier (\"0,1\"; default: every visible GPU)")
    ap.add_argument("--tokenizer", default=str(ROOT / "pack/full/tokenizer"),
                    help="tokenizer directory for the mock engine (a byte tokenizer if absent)")
    ap.add_argument("--open", action="store_true", help="open the local page in the browser once the model is ready")
    ap.add_argument("--fit-max-tokens", action="store_true",
                    help="clamp max_tokens to the remaining context instead of rejecting the request with 400")
    ap.add_argument("--api-key", default="",
                    help="require this key on /v1/* (Authorization: Bearer ... or x-api-key); command line only, "
                         "not allowed in a --config file")
    ap.add_argument("--allowed-host", action="append", default=[], metavar="NAME",
                    help="also answer to this Host name (repeatable; \"*\" turns the check off)")
    ap.add_argument("--cors-origin", action="append", default=[], metavar="ORIGIN",
                    help="let web pages from this origin use /v1/* (repeatable)")
    ap.add_argument("--debug", action="store_true",
                    help="log one line per request and the raw model text after each reply")
    ap.add_argument("--request-lines", action="store_true",
                    help="echo the engine's per-request summary on stdout, for a supervisor that only sees this output")
    ap.add_argument("--lazy", action="store_true", help="start the text-only API unloaded; load on first request")
    ap.add_argument("--idle-unload", type=float, default=None, metavar="SECONDS",
                    help="unload the model after this many seconds without requests; the next request loads it again")
    ap.add_argument("--min-free-vram-mib", type=int, default=None,
                    help="load an unloaded model only when this much VRAM is free, else answer 503")
    return ap
