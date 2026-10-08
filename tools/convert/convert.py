"""Build the single .basalt model file from the source GGUF shards.

    python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --out MODEL.basalt \\
        [--mtp-source DIR] [--mtp-dense FILE] [--profile FILE | --profile-from TRACE ...] [--draft-vocab FILE] [--hc8 FILE] \\
        [--name NAME] [--compat-bf16] [--no-vision] [--force]

Inputs
  --gguf FILE        the model's GGUF: shard 1 of a split model (the others are found beside it) or --gguf per shard
  --mmproj FILE      the vision projector GGUF (required unless --no-vision)
  --mtp-source DIR   the draft layer's BF16 tensors from `mtp_fetch.py fetch --out DIR`; without it there is no drafter
  --mtp-dense FILE   a tuned `mtp/dense` section (raw bytes, same layout as the source; needs --mtp-source); the packs
                     with a self-distilled MTP layer carry one. Its size is checked and its SHA-256 goes to meta.mtp
  --draft-vocab FILE the token subset the draft head may propose (int32 ids; needs --mtp-source)
  --profile FILE     an expert profile stored as is (`--expert-profile-save`'s file)
  --profile-from TRACE ...  `--dump-routing` traces: routed experts first, most frequent first, then the rest;
                     with neither option the ranking is the plain fill order
  --hc8 FILE         8-bit hyper-connection copies from `--dump-hc8`; without them the engine quantizes at every start

Options
  --name NAME        the model's name, reported by the server
  --compat-bf16      convert small projections stored in another form to BF16 (round-to-nearest-even, recorded in
                     meta.conversions)
  --no-vision        leave out the vision projector
  --force            overwrite --out

Sections are listed in include/basalt/core/model_format.hpp. Everything is streamed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))   # for model_file, gguf_reader, tokenizer
import model_sections as S  # noqa: E402
from model_file import TENSOR_ALIGN, ModelWriter  # noqa: E402

FINGERPRINT_BYTES = 64 << 20

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
LICENSE_TEXT = (HERE / "qwen-community-license-1.0.txt").read_text(encoding="utf-8")


def model_license(kv: dict) -> dict:
    """The model's license, with its full text (the license requires it in every copy)."""
    lic = {"name": "Qwen Community License 1.0", "text": LICENSE_TEXT}
    if kv.get("general.quantized_by"):
        lic["quantized_by"] = kv["general.quantized_by"]
    return lic


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


def load_vision_helper():
    import pack_vision
    return pack_vision.vision_sections


def export_tokenizer(first_gguf: Path, out_dir: Path) -> list[Path]:
    """The tokenizer files (tools/tokenizer.py) of the model's first shard, written under `out_dir`/tokenizer."""
    import tokenizer
    tokenizer.extract(first_gguf, out_dir)
    return sorted(p for p in (out_dir / "tokenizer").iterdir() if p.is_file())


def source_tensors(model: S.SourceModel):
    """(kv, [(name, shard path, absolute offset, size, TensorInfo)]) over every shard, expert tensors excluded."""
    kv, tensors = {}, []
    for p, g in zip(model.paths, model.files):
        for k, v in g.metadata.items():
            kv.setdefault(k, json_safe(v))
    for name, (g, t, p) in model.where.items():
        if S.is_expert(name):
            continue
        a, n = g.data_start + t.offset, t.expected_bytes()
        tensors.append((name, p, a, n, t))
    order = {p: i for i, p in enumerate(model.paths)}
    tensors.sort(key=lambda x: (order[x[1]], x[2]))      # sequential reads per shard
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
        self._report(name, self.w.entries[-1][2], time.time() - t)

    def chunks(self, name, chunks):
        t = time.time()
        n = self.w.add_chunks(name, chunks)
        self._report(name, n, time.time() - t)


def source_paths(given) -> list[Path]:
    paths = [Path(p) for p in given]
    for p in paths:
        if not p.is_file():
            die(f"missing input: {p}")
    return S.expand_shards(paths[0]) if len(paths) == 1 else paths


def convert(a) -> None:
    out = Path(a.out)
    partial = Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    mmproj = Path(a.mmproj) if a.mmproj else None
    side = [Path(x) for x in (a.hc8, a.draft_vocab, a.profile, a.mtp_dense) if x] + [Path(t) for t in (a.profile_from or [])]
    if not a.no_vision:
        side.append(mmproj if mmproj else Path("<--mmproj>"))
    for p in side:
        if not p.is_file():
            die(f"missing input: {p}")
    if a.draft_vocab and not a.mtp_source:
        die("--draft-vocab needs --mtp-source")
    if a.mtp_dense and not a.mtp_source:
        die("--mtp-dense needs --mtp-source")
    if a.profile and a.profile_from:
        die("give --profile or --profile-from, not both")

    try:
        print("reading the source GGUFs ...", flush=True)
        model = S.SourceModel(source_paths(a.gguf))
        kv, tensors = source_tensors(model)
        dense = S.Dense(model, a.compat_bf16)
        experts = S.Experts(model)
        mtp = S.Mtp(a.mtp_source, a.mtp_dense) if a.mtp_source else None
        n_vocab = len(kv.get("tokenizer.ggml.tokens") or []) or None
        draft_vocab = S.draft_vocab_bytes(a.draft_vocab, n_vocab) if a.draft_vocab else None
        if a.profile:
            profile_blob = Path(a.profile).read_bytes()
        else:
            freq = {}
            for t in a.profile_from or []:
                for p, c in S.read_trace(t, experts.n_layers, experts.n_expert).items():
                    freq[p] = freq.get(p, 0) + c
            if not a.profile_from:
                print("note: no profile and no routing traces: the ranking is the plain fill order", flush=True)
            profile_blob = S.build_profile(S.rank_pairs(freq, experts.n_layers, experts.n_expert),
                                           experts.n_layers, experts.n_expert)
        prof = S.profile_info(profile_blob, experts.n_layers, experts.n_expert)
    except S.ConvertError as e:
        die(str(e))

    vision_extra = []
    if not a.no_vision:
        fn = load_vision_helper()
        if fn is None:
            print("note: pack_vision.py (vision_sections) not found; only vision/mmproj is written")
        else:
            vision_extra = list(fn(str(mmproj), [str(p) for p in model.paths]))

    if out.parent and not out.parent.exists():
        die(f"{out.parent} does not exist")
    if partial.exists():
        partial.unlink()
    w = ModelWriter(str(partial))
    job = Job(w)
    ok = False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            print("writing sections:", flush=True)
            for p in export_tokenizer(model.paths[0], Path(tmp)):
                job.file(f"tokenizer/{p.name}", p)
            job.chunks("dense", dense.chunks())
            job.bytes("dense.index", dense.index_text())
            job.chunks("experts", experts.chunks())
            job.bytes("experts.layout", experts.layout_text())
            t = time.time()
            offs = w.add_parts("gguf", [(src, off, size) for _, src, off, size, _ in tensors], TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            tensor_offsets = dict(zip((x[0] for x in tensors), offs))
            if mtp is not None:
                job.chunks("mtp/dense", mtp.dense_chunks())
                job.bytes("mtp/dense.index", mtp.dense_index())
                job.chunks("mtp/experts", mtp.expert_chunks())
                if draft_vocab is not None:
                    job.bytes("mtp/draft_vocab", draft_vocab)
            job.bytes("profile", profile_blob)
            if a.hc8:
                job.file("hc8", a.hc8)
            written = {e[0] for e in w.entries}
            if not a.no_vision:
                job.file("vision/mmproj", mmproj)
                for name, data, src, off, size in vision_extra:
                    if name in written or name == "vision/mmproj":
                        print(f"note: vision section {name} already written, skipped")
                        continue
                    if data is not None:
                        job.bytes(name, data)
                    else:
                        job.file(name, src, off, size)

            sources = [{"path": p.name, "size": p.stat().st_size,
                        "fingerprint_sha256_first_last_64mib_and_size": fingerprint(p)}
                       for p in model.paths + ([mmproj] if not a.no_vision else [])]
            meta = {
                "format": "basalt", "version": 1, "name": a.name,
                "converter": {"tool": "tools/convert/convert.py", "git": git_rev()},
                "sources": sources,
                "gguf_kv": kv,
                "tensors": {n: {"type": t.type_name, "type_id": t.type_id, "shape": [int(d) for d in t.shape],
                                "bytes": nb, "offset": tensor_offsets[n]}
                            for n, _, _, nb, t in tensors},
                "conversions": {"compat_bf16": a.compat_bf16, "tensors": dense.records},
                "sampling": SAMPLING, "license": model_license(kv),
                "profile": prof,
                "hc8": {"bytes": Path(a.hc8).stat().st_size} if a.hc8 else None,
                "mtp": None if mtp is None else {"source_sha256": mtp.source_sha256(),
                                                  "expert_format": "q2_0", "draft_vocab": draft_vocab is not None,
                                                  "dense": ({"tuning": "self_distilled", "sha256": mtp.tuned_sha256}
                                                            if mtp.tuned else {"tuning": None})},
                "vision": None if a.no_vision else {"mmproj": "vision/mmproj", "mmproj_bytes": mmproj.stat().st_size,
                                                    "extra": [v[0] for v in vision_extra]},
            }
            job.bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        w.close()
        ok = True
    except S.ConvertError as e:
        die(str(e))
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, action="append", metavar="FILE",
                    help="the model's GGUF: shard 1 of a split model (the others are found beside it), or repeat per shard")
    ap.add_argument("--mmproj", help="vision projector GGUF (required unless --no-vision)")
    ap.add_argument("--out", required=True, help="the .basalt file to write")
    ap.add_argument("--mtp-source", metavar="DIR", help="the drafter's BF16 tensors written by tools/convert/mtp_fetch.py fetch")
    ap.add_argument("--mtp-dense", metavar="FILE", help="a tuned (self-distilled) mtp/dense section to embed (needs --mtp-source)")
    ap.add_argument("--draft-vocab", metavar="FILE", help="int32 token ids the draft head may propose (needs --mtp-source)")
    ap.add_argument("--profile", metavar="FILE", help="an expert profile to store as it is (--expert-profile-save's file)")
    ap.add_argument("--profile-from", nargs="+", metavar="TRACE",
                    help="routing traces from `basalt generate --dump-routing`: the ranking is built from them")
    ap.add_argument("--hc8", metavar="FILE", help="file written by `basalt generate --model ... --dump-hc8`")
    ap.add_argument("--name", default="Qwen3.8-Flash-Next IQ3_XXS")
    ap.add_argument("--compat-bf16", action="store_true",
                    help="convert small projections stored in another form to BF16 (rounds)")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    a = ap.parse_args()
    if not a.no_vision and not a.mmproj:
        ap.error("--mmproj is required unless --no-vision")
    convert(a)


if __name__ == "__main__":
    main()
