#!/usr/bin/env python3
"""tools/ppl_gate.py - perplexity and KL divergence through the decode path, against a llama.cpp base.

The base is the file `llama-perplexity --kl-divergence-base` writes (wikitext-2, n_ctx 512 chunks, the second half
of each chunk scored).  The engine is started the way serve/server.py starts it from a config (the vision encoder
first, so the expert tiers are sized as in daily use), with STRATA_PPL_BASE/STRATA_PPL_OUT set and any STRATA_*
flags of the caller's environment.  Per chunk it reads the first half as a prompt and teacher-forces the rest
through verify windows of T tokens (every token committed); see include/strata/program/ppl_gate.hpp.

  python3 tools/ppl_gate.py --exe build/strata --config cfg.json --base base.kld --out DIR [--chunks 20] [--t 3]
                            [--set="--spec 5 --spec-min-p 0.7"]
  python3 tools/ppl_gate.py --report DIR/rows.tsv          # the summary of a finished run again
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path


def read_base(path: str):
    with open(path, "rb") as f:
        if f.read(8) != b"_logits_":
            sys.exit(f"{path}: not a llama.cpp KL-divergence base")
        n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", f.read(12))
        toks = struct.unpack(f"<{n_ctx * n_chunk}i", f.read(4 * n_ctx * n_chunk))
    return n_ctx, n_vocab, n_chunk, [toks[k * n_ctx:(k + 1) * n_ctx] for k in range(n_chunk)]


def summary(rows_path: str, extra: dict | None = None) -> str:
    nll, nllb, kld, same, dlp = [], [], [], 0, []
    chunks = set()
    for line in open(rows_path):
        f = line.split("\t")
        if len(f) < 10:
            continue
        chunks.add(int(f[0]))
        nll.append(float(f[3])); nllb.append(float(f[4])); kld.append(float(f[5]))
        same += int(f[6]); dlp.append(float(f[9]))
    n = len(nll)
    if n == 0:
        return "no rows"
    ks = sorted(kld)
    pct = lambda q: ks[min(n - 1, int(round(q * (n - 1))))]
    mn = sum(nll) / n
    se = math.sqrt(max(0.0, sum(v * v for v in nll) / n - mn * mn) / max(1, n - 1))
    out = [f"chunks {len(chunks)}  rows {n}",
           f"PPL {math.exp(mn):.4f} +/- {math.exp(mn) * se:.4f}   base PPL {math.exp(sum(nllb) / n):.4f}   "
           f"ln(PPL/base) {mn - sum(nllb) / n:.5f}",
           f"KLD mean {sum(kld) / n:.5f}  median {pct(0.5):.5f}  p99 {pct(0.99):.4f}  max {ks[-1]:.4f}",
           f"same top-1 {100.0 * same / n:.2f}%   mean |dlogp| of the base top-1 {sum(dlp) / n:.5f}"]
    if extra:
        out.append("  ".join(f"{k} {v}" for k, v in extra.items()))
    return "\n".join(out)


def engine_env(cfg: dict) -> dict:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    return env


def start_vision(cfg: dict, env: dict, log):
    v = cfg.get("vision")
    if not v:
        return None
    args = [v["exe"], "--mmproj", v["mmproj"], "--model", v["model"]]
    if v.get("gpu"):
        args.append("--gpu")
    if v.get("threads"):
        args += ["--threads", str(v["threads"])]
    if v.get("max_tokens"):
        args += ["--max-tokens", str(v["max_tokens"])]
    venv = dict(env)
    if v.get("cuda_device") is not None:   # serve/server.py vision_env
        venv["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        venv["CUDA_VISIBLE_DEVICES"] = str(v["cuda_device"])
    for k in [k for k in venv if k.startswith("STRATA_PPL")]:
        del venv[k]
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=venv)
    line = p.stdout.readline()
    if not line.startswith("READY"):
        sys.exit("the vision encoder did not start: " + line.strip())
    return p


def run(a) -> int:
    n_ctx, n_vocab, n_chunk, chunks = read_base(a.base)
    first = n_ctx // 2
    cfg = json.load(open(a.config))
    args = list(cfg["args"])
    for k in range(0, len(a.set), 2):   # --set FLAG VALUE ...: replace or add engine arguments
        if a.set[k] in args:
            args[args.index(a.set[k]) + 1] = a.set[k + 1]
        else:
            args += [a.set[k], a.set[k + 1]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = out / "rows.tsv"
    rows.unlink(missing_ok=True)
    env = engine_env(cfg)
    env["STRATA_PPL_BASE"] = a.base
    env["STRATA_PPL_OUT"] = str(rows)
    log = open(out / "engine.log", "w")
    flags = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("STRATA_"))
    log.write(f"ppl_gate: exe={a.exe} T={a.t} chunks={a.chunks} args={' '.join(args)}\nenv: {flags}\n")
    log.flush()
    t0 = time.time()
    vis = start_vision(cfg, env, log)
    eng = subprocess.Popen([a.exe or cfg["exe"], "--serve", *args], cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=env)
    try:
        for line in eng.stdout:
            if line.startswith("READY"):
                break
        else:
            print("the engine exited before READY; see", out / "engine.log")
            return 1
        t_load = time.time() - t0
        hits = look = off = 0
        pp_ms = dec_ms = 0.0
        nc = min(a.chunks, n_chunk) if a.chunks > 0 else n_chunk
        t1 = time.time()
        for k in range(nc):
            ids = ",".join(str(t) for t in chunks[k][:first + 1])
            eng.stdin.write(f"GEN {n_ctx - 1 - first} ppl={k} ppl_t={a.t} {ids}\n")
            eng.stdin.flush()
            for line in eng.stdout:
                if line.startswith("ERR"):
                    print(f"chunk {k}: {line.strip()}")
                    return 1
                if line.startswith("DONE"):
                    f = line.split()
                    pp_ms += float(f[3]); dec_ms += float(f[4])
                    if f[8] != "0":
                        print(f"chunk {k}: {f[8]} prompt tokens reused from an earlier request")
                    if len(f) > 10:
                        hits += int(f[9]); look += int(f[10])
                    if len(f) > 15:
                        off += int(f[15])
                    break
            else:
                print(f"chunk {k}: the engine exited; see", out / "engine.log")
                return 1
            if a.progress and (k + 1) % a.progress == 0:
                print(f"  {k + 1}/{nc} chunks, {time.time() - t1:.0f} s", flush=True)
        t_run = time.time() - t1
        eng.stdin.write("QUIT\n")
        eng.stdin.flush()
    finally:
        try:
            eng.wait(timeout=60)
        except subprocess.TimeoutExpired:
            eng.kill()
        if vis is not None:
            vis.kill()
    extra = {"T": a.t, "load s": f"{t_load:.0f}", "run s": f"{t_run:.1f}",
             "prompt ms/chunk": f"{pp_ms / nc:.0f}", "decode ms/chunk": f"{dec_ms / nc:.0f}",
             "decode VRAM-cache hits": f"{100.0 * hits / look:.1f}% ({hits}/{look})" if look else "n/a",
             "offloaded": off}
    rep = summary(str(rows), extra)
    (out / "report.txt").write_text(f"exe {a.exe}\nenv {flags}\n{rep}\n")
    print(rep)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exe")
    ap.add_argument("--config")
    ap.add_argument("--base")
    ap.add_argument("--out")
    ap.add_argument("--chunks", type=int, default=0, help="the first N chunks (0: all)")
    ap.add_argument("--t", type=int, default=3, help="tokens per verify window")
    ap.add_argument("--set", default="", help='"FLAG VALUE ..." engine arguments to replace or add (--set="--spec 5")')
    ap.add_argument("--progress", type=int, default=10, help="a progress line every N chunks (0: none)")
    ap.add_argument("--report", help="print the summary of an existing rows.tsv and exit")
    a = ap.parse_args()
    if a.report:
        print(summary(a.report))
        return 0
    if not (a.config and a.base and a.out):
        ap.error("--config, --base and --out are required")
    a.set = a.set.split()
    if len(a.set) % 2:
        ap.error("--set takes FLAG VALUE pairs")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
