"""tools/convert.py - assemble the single .basalt model file from a pack, the source GGUFs and the side files.

    python3 tools/convert.py --pack DIR --native SHARD1.gguf [--native SHARD2.gguf ...] --mtp DIR --profile FILE \\
        --hc8 FILE --mmproj FILE --out MODEL.basalt [--name NAME] [--no-vision] [--force]

The sections written are listed in include/basalt/core/model_format.hpp.  Large inputs are streamed (nothing is
loaded whole); GGUF tensors are copied straight out of the shards by offset.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gguf_reader import GGUFFile  # noqa: E402
from model_file import TENSOR_ALIGN, ModelWriter  # noqa: E402

EXPERT_RE = re.compile(r"^blk\.\d+\.ffn_(gate|up|down|gate_up)_exps\.weight$")
FINGERPRINT_BYTES = 64 << 20
BRAND = "strata"

ENGINE_ARGS = ["--expert-cache", "auto", "--prefill", "auto", "--spec", "7", "--spec-min-p", "0.5",
               "--max-context", "262144", "--kv", "int8", "--resident-experts", "--vram-reserve-mib", "700"]
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
LICENSE = {"model": "Qwen Community License (Qwen3.8-Flash-Next) and Apache-2.0 (the GSQ-RCO GGUF); see the model card",
           "engine": "Basalt, MIT; based on Strata (MIT)"}
PROFILE_MAGIC = b"STRP"
PROFILE_HEADER = struct.Struct("<5I")


def die(msg: str):
    raise SystemExit(f"convert: {msg}")


# ------------------------------------------------------------------ helpers
def fingerprint(path: Path) -> str:
    """sha256 over the first and last 64 MiB and the size: fast, and enough to tell two files apart."""
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(FINGERPRINT_BYTES))
        f.seek(max(0, size - FINGERPRINT_BYTES))
        h.update(f.read(FINGERPRINT_BYTES))
    h.update(struct.pack("<Q", size))
    return h.hexdigest()


def git_rev():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except OSError:
        return None


def json_safe(v):
    """GGUF metadata as JSON: non-finite floats become strings, which JSON cannot carry."""
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    if isinstance(v, list):
        return [json_safe(x) for x in v]
    return v


def replace_first_line(data: bytes, new_line: bytes) -> bytes:
    nl = data.find(b"\n")
    return new_line + (b"" if nl < 0 else data[nl:])


def rewrite_index_header(data: bytes) -> bytes:
    first = data.split(b"\n", 1)[0].decode("utf-8", "replace")
    if not re.match(r"^# strata pack index v\d+", first):
        die(f"dense index: unexpected first line {first[:80]!r}")
    if first.split()[4] != "v3":
        die(f"dense index: version {first.split()[4]}, this converter writes v3")
    return replace_first_line(data, b"# dense index v3")


def rewrite_layout_header(data: bytes) -> bytes:
    first, _, rest = data.partition(b"\n")
    m = re.match(rb"^# strata native experts (v\d+)", first)
    if not m:
        die(f"experts layout: unexpected first line {first[:80]!r}")
    return b"# experts " + m.group(1) + first[m.end():] + b"\n" + rest


def rewrite_mtp_index(data: bytes) -> bytes:
    """The mtp dense index has no header today; if one with a brand ever appears, it is made neutral."""
    first = data.split(b"\n", 1)[0]
    if not first.lower().startswith(b"# strata"):
        return data
    ver = re.search(rb"\bv\d+\b", first)
    line = b"# mtp dense index" + (b" " + ver.group(0) if ver else b"")
    return replace_first_line(data, line)


def check_unbranded(what: str, data: bytes):
    if BRAND.encode() in data.lower():
        die(f"{what}: contains the string '{BRAND}'")


def profile_info(blob: bytes) -> dict:
    """The profile without its magic (see tools/make_profile.py) and a description of what remains."""
    if blob[:4] != PROFILE_MAGIC:
        die("profile: not an expert profile (no STRP magic)")
    ver, nl, ne, slots, n = PROFILE_HEADER.unpack_from(blob, 4)
    want = 4 + PROFILE_HEADER.size + 4 * n + 4 * nl * ne
    if len(blob) != want:
        die(f"profile: {len(blob)} bytes, expected {want} for {n} pairs of {nl}x{ne}")
    pairs_off = PROFILE_HEADER.size
    return {"magic_stripped": PROFILE_MAGIC.decode(), "version": ver, "n_layer": nl, "n_expert": ne,
            "slots": slots, "n_pairs": n,
            "layout": "5 x u32 LE (version, n_layer, n_expert, slots, n_pairs); then pairs: n_pairs x (u16 layer, "
                      "u16 expert) in rank order; then table: n_layer x n_expert x i32 LE, the rank of each "
                      "(layer, expert) or -1",
            "pairs_offset": pairs_off, "table_offset": pairs_off + 4 * n}


def load_vision_helper():
    path = HERE / "vision" / "pack_vision.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("pack_vision", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "vision_sections", None)


# ------------------------------------------------------------------ source GGUFs
def read_ggufs(paths):
    """(kv, [(name, shard path, abs offset, size, TensorInfo)]) over every shard, expert tensors excluded."""
    kv, tensors = {}, []
    for p in paths:
        g = GGUFFile(p)
        for k, v in g.metadata.items():
            kv.setdefault(k, json_safe(v))
        size = p.stat().st_size
        order = sorted(g.tensors, key=lambda t: t.offset)
        for i, t in enumerate(order):
            if EXPERT_RE.match(t.name):
                continue
            nbytes = t.expected_bytes()
            if nbytes is None:
                end = order[i + 1].offset if i + 1 < len(order) else size - g.data_start
                nbytes = end - t.offset
            a = g.data_start + t.offset
            if a + nbytes > size:
                die(f"{p.name}: tensor {t.name} runs past the end of the file")
            tensors.append((t.name, p, a, nbytes, t))
    names = [t[0] for t in tensors]
    if len(set(names)) != len(names):
        die("a tensor name appears in more than one shard")
    return kv, tensors


# ------------------------------------------------------------------ the conversion
class Job:
    def __init__(self, writer: ModelWriter):
        self.w = writer
        self.rows: list[tuple[str, int]] = []
        self.t0 = time.time()

    def _report(self, name, nbytes, secs):
        mbs = nbytes / (1 << 20) / secs if secs > 0 else float("inf")
        print(f"  {name:<44} {nbytes / (1 << 30):9.3f} GiB {mbs:9.0f} MB/s", flush=True)
        self.rows.append((name, nbytes))

    def bytes(self, name, data: bytes):
        t = time.time()
        self.w.add_bytes(name, data)
        self._report(name, len(data), time.time() - t)

    def file(self, name, src, offset=0, size=None):
        t = time.time()
        self.w.add_file(name, str(src), offset, size)
        n = self.w.entries[-1][2]
        self._report(name, n, time.time() - t)


def convert(a) -> None:
    out = Path(a.out)
    partial = Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    pack, mtp = Path(a.pack), Path(a.mtp)
    natives = [Path(p) for p in a.native]
    need = [pack / "dense.bin", pack / "index.txt", pack / "experts.bin", pack / "native_experts.txt",
            mtp / "dense.bin", mtp / "dense.txt", mtp / "experts.bin", mtp / "draft_vocab.bin",
            Path(a.profile), Path(a.hc8), *natives]
    if not a.no_vision:
        need.append(Path(a.mmproj) if a.mmproj else Path("<--mmproj>"))
    for p in need:
        if not p.is_file():
            die(f"missing input: {p}")
    tok_dir = pack / "tokenizer"
    tok_files = sorted(p for p in tok_dir.iterdir() if p.is_file()) if tok_dir.is_dir() else []
    if not tok_files:
        die(f"no tokenizer files in {tok_dir}")

    print("reading the source GGUFs ...", flush=True)
    kv, tensors = read_ggufs(natives)
    profile_blob = Path(a.profile).read_bytes()
    prof = profile_info(profile_blob)

    vision_extra = []
    if not a.no_vision:
        fn = load_vision_helper()
        if fn is None:
            print("note: tools/vision/pack_vision.py (vision_sections) not found; only vision/mmproj is written")
        else:
            vision_extra = list(fn(str(a.mmproj), [str(p) for p in natives]))

    if out.parent and not out.parent.exists():
        die(f"{out.parent} does not exist")
    if partial.exists():
        partial.unlink()
    w = ModelWriter(str(partial))
    job = Job(w)
    ok = False
    try:
        print("writing sections:", flush=True)
        for p in tok_files:
            job.file(f"tokenizer/{p.name}", p)
        job.file("dense", pack / "dense.bin")
        idx = rewrite_index_header((pack / "index.txt").read_bytes())
        check_unbranded("dense.index", idx.split(b"\n", 1)[0])
        job.bytes("dense.index", idx)
        job.file("experts", pack / "experts.bin")
        lay = rewrite_layout_header((pack / "native_experts.txt").read_bytes())
        check_unbranded("experts.layout header", lay.split(b"\n", 1)[0])
        job.bytes("experts.layout", lay)
        t = time.time()
        offs = w.add_parts("gguf", [(src, off, size) for _, src, off, size, _ in tensors], TENSOR_ALIGN)
        job._report("gguf", w.entries[-1][2], time.time() - t)
        tensor_offsets = dict(zip((x[0] for x in tensors), offs))
        job.file("mtp/dense", mtp / "dense.bin")
        job.bytes("mtp/dense.index", rewrite_mtp_index((mtp / "dense.txt").read_bytes()))
        job.file("mtp/experts", mtp / "experts.bin")
        job.file("mtp/draft_vocab", mtp / "draft_vocab.bin")
        job.bytes("profile", profile_blob[4:])
        job.file("hc8", a.hc8)
        written = {e[0] for e in w.entries}
        if not a.no_vision:
            job.file("vision/mmproj", a.mmproj)
            for name, data, src, off, size in vision_extra:
                if name in written or name == "vision/mmproj":
                    print(f"note: vision section {name} already written, skipped")
                    continue
                if data is not None:
                    job.bytes(name, data)
                else:
                    job.file(name, src, off, size)

        sources = [{"path": p.name, "size": p.stat().st_size, "fingerprint_sha256_first_last_64mib_and_size": fingerprint(p)}
                   for p in natives + ([Path(a.mmproj)] if not a.no_vision else [])]
        meta = {
            "format": "basalt", "version": 1, "name": a.name,
            "converter": {"tool": "tools/convert.py", "git": git_rev()},
            "sources": sources,
            "gguf_kv": kv,
            "tensors": {n: {"type": t.type_name, "type_id": t.type_id, "shape": [int(d) for d in t.shape], "bytes": nb,
                            "offset": tensor_offsets[n]}
                        for n, _, _, nb, t in tensors},
            "engine_args": ENGINE_ARGS, "sampling": SAMPLING, "license": LICENSE,
            "profile": prof,
            "hc8": {"bytes": Path(a.hc8).stat().st_size},
            "vision": None if a.no_vision else {"mmproj": "vision/mmproj", "mmproj_bytes": Path(a.mmproj).stat().st_size,
                                                "extra": [v[0] for v in vision_extra]},
        }
        rest = dict(meta)
        rest.pop("gguf_kv")
        rest.pop("license")                 # the attribution names the upstream project
        check_unbranded("meta", json.dumps(rest).encode())
        if BRAND.encode() in json.dumps(kv).lower().encode():
            print(f"warning: the source GGUF metadata mentions '{BRAND}' (copied verbatim into meta.gguf_kv)")
        job.bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        w.close()
        ok = True
    finally:
        if not ok:
            try:
                w.f.close()
            except OSError:
                pass
            partial.unlink(missing_ok=True)
    os.replace(partial, out)
    total = out.stat().st_size
    print(f"done: {len(job.rows)} sections, {total / (1 << 30):.3f} GiB, {time.time() - job.t0:.1f} s -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True, help="pack directory (dense.bin, index.txt, experts.bin, ...)")
    ap.add_argument("--native", required=True, action="append", help="source GGUF shard (repeat per shard)")
    ap.add_argument("--mtp", required=True, help="mtp runtime directory")
    ap.add_argument("--profile", required=True, help="expert profile (STRP)")
    ap.add_argument("--hc8", required=True, help="file written by `basalt --dump-hc8`")
    ap.add_argument("--mmproj", help="vision projector GGUF (required unless --no-vision)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="Qwen3.8-Flash-Next IQ3_XXS")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    a = ap.parse_args()
    if not a.no_vision and not a.mmproj:
        ap.error("--mmproj is required unless --no-vision")
    convert(a)


if __name__ == "__main__":
    main()
