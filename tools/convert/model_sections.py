"""Sections of a .basalt file built from the source weights, for tools/convert/convert.py.

  dense / dense.index       float tensors in the form the engine reads, and the table of every non-expert tensor
  experts / experts.layout  per layer, n_expert blobs of [gate | up | down] rows, and where each layer starts
  mtp/dense, mtp/dense.index, mtp/experts  the draft layer from the BF16 `mtp.*` tensors (tools/convert/mtp_fetch.py)
  profile                   the expert ranking from routing traces or an existing profile

Sections are produced as byte-chunk iterators for ModelWriter.add_chunks. Needs numpy.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import struct
import sys
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ggufq  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402


class ConvertError(Exception):
    """An input that cannot be converted; the message says why."""


FLOAT = {"BF16", "F32", "F16"}
ROLES = ("gate", "up", "down")
ALIGN = 64                      # between the tensors of `dense`
EXPERT_RE = re.compile(r"^blk\.\d+\.ffn_(gate|up|down|gate_up)_exps\.weight$")
# the PLE table comes from the `gguf` section, never the dense pool
NOT_IN_DENSE = {"per_layer_token_embd.weight"}

# float form the engine reads each dense tensor in (name without `blk.N.`); other forms are converted and recorded
# in meta.conversions:
#   F32 -> BF16 when exactly BF16 (low 16 bits zero)              exact, by default
#   other F32, F16 or quantized -> BF16                           rounds (nearest-even): only with --compat-bf16
#   F32 -> F16 (ple_conv1d)                                       index kind 3, narrowed by the loader
#   F16 / BF16 -> F32                                             exact widening
# tensors not listed are written as stored (floats) or served from the `gguf` section (quantized)
FORM = {
    "ffn_gate_inp.weight": "BF16", "ffn_gate_inp_shexp.weight": "BF16",
    "hc_attn_down.weight": "BF16", "hc_attn_up.weight": "BF16", "hc_attn_inject.weight": "BF16",
    "hc_ffn_down.weight": "BF16", "hc_ffn_up.weight": "BF16", "hc_ffn_inject.weight": "BF16",
    "output_hc_down.weight": "BF16", "output_hc_up.weight": "BF16",
    "indexer.k_proj.weight": "BF16", "indexer.q_proj.weight": "BF16",
    "ssm_alpha.weight": "BF16", "ssm_beta.weight": "BF16",
    "ple_key.weight": "BF16", "ple_value.weight": "BF16", "ple_conv1d.weight": "F16",
    "attn_q_norm.weight": "F32", "attn_k_norm.weight": "F32", "hc_attn_norm.weight": "F32",
    "hc_ffn_norm.weight": "F32", "output_hc_norm.weight": "F32", "indexer.q_norm.weight": "F32",
    "indexer.k_norm.weight": "F32", "ple_norm_conv.weight": "F32", "ple_norm_key.weight": "F32",
    "ple_norm_query.weight": "F32", "ssm_a": "F32", "ssm_conv1d.weight": "F32", "ssm_dt.bias": "F32",
    "ssm_norm.weight": "F32",
}
# PLE key encodings the engine reads natively; other quantized keys need --compat-bf16
NATIVE_PLE_KEY = {"Q2_0", "Q8_0"}
KIND = {"BF16": "4", "F16": "5", "F32": "2"}


def form_of(name: str):
    return FORM.get(re.sub(r"^blk\.\d+\.", "", name))


# ------------------------------------------------------------------ the source model
def expand_shards(first: pathlib.Path) -> list[pathlib.Path]:
    """`<name>-00001-of-0000N.gguf` -> all N shards (a missing one is an error); any other name -> itself."""
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
    if not m:
        return [first]
    total = int(m.group(2))
    paths = [first.with_name(first.name[:m.start()] + "-%05d-of-%05d.gguf" % (i, total)) for i in range(1, total + 1)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ConvertError("missing model shards: " + ", ".join(missing))
    return paths


def check_split(files) -> None:
    """The split keys as the engine checks them: shard 1 carries the metadata and every shard declares split.count / split.no consistently."""
    n = len(files)
    meta0 = files[0].metadata
    if n == 1:
        if int(meta0.get("split.count", 1)) > 1:
            raise ConvertError(f"{files[0].path.name} is shard 1 of {meta0['split.count']}, but its name has no "
                               "-00001-of-0000N.gguf to find the others by")
        return
    if "general.architecture" not in meta0:
        raise ConvertError(f"{files[0].path.name} has no general.architecture; the first shard of a split model "
                           "carries the metadata")
    total = meta0.get("split.tensors.count")
    for i, g in enumerate(files):
        md = g.metadata
        if md.get("split.count") != n or md.get("split.no") != i or \
                (total is not None and md.get("split.tensors.count") != total):
            raise ConvertError(f"{g.path.name} does not declare itself shard {i + 1} of {n} of this model "
                               "(split.count / split.no / split.tensors.count)")
    if total is not None and sum(len(g.tensors) for g in files) != total:
        raise ConvertError(f"the {n} shards hold {sum(len(g.tensors) for g in files)} tensors, but "
                           f"split.tensors.count is {total}")


class SourceModel:
    """All shards of one model: name -> (GGUFFile, TensorInfo, shard path), in file order."""

    def __init__(self, paths):
        self.paths = [pathlib.Path(p) for p in paths]
        self.files = [GGUFFile(p) for p in self.paths]
        self.sizes = [p.stat().st_size for p in self.paths]
        check_split(self.files)
        self._maps: dict[pathlib.Path, np.ndarray] = {}
        self.where: dict[str, tuple] = {}
        for p, g, size in zip(self.paths, self.files, self.sizes):
            for t in g.tensors:
                n = t.expected_bytes()
                if n is None or g.data_start + t.offset + n > size:
                    raise ConvertError(f"{p.name}: unsupported or truncated tensor {t.name}")
                if t.name in self.where:
                    raise ConvertError(f"{p.name}: duplicate tensor {t.name}")
                self.where[t.name] = (g, t, p)

    def bytes(self, name: str) -> np.ndarray:
        g, t, p = self.where[name]
        if p not in self._maps:
            self._maps[p] = np.memmap(p, dtype=np.uint8, mode="r")
        a = g.data_start + t.offset
        return self._maps[p][a:a + t.expected_bytes()]

    def abs_range(self, name: str):
        """(shard path, absolute offset, size) of a tensor's bytes."""
        g, t, p = self.where[name]
        return p, g.data_start + t.offset, t.expected_bytes()


def is_expert(name: str) -> bool:
    return bool(EXPERT_RE.match(name))


# ------------------------------------------------------------------ dense
def _dequantize(raw: np.ndarray, type_name: str) -> np.ndarray:
    """The tensor's values as float32 (floats exactly, quantized types through tools/ggufq.py)."""
    if type_name == "F32":
        return raw.view(np.float32)
    if type_name == "F16":
        return raw.view(np.float16).astype(np.float32)
    if type_name == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return ggufq.dequantize(raw, ggufq.GGMLQuantizationType[type_name]).astype(np.float32).reshape(-1)


def bf16_bytes(raw: np.ndarray, type_name: str) -> bytes:
    """Round-to-nearest-even BF16 of the tensor's values (ggml's conversion, halfway cases included)."""
    values = ggufq.dequantize(raw, ggufq.GGMLQuantizationType[type_name])
    if not np.isfinite(values).all():
        raise ValueError("cannot convert non-finite weights to BF16")
    return ggufq.quantize(values, ggufq.GGMLQuantizationType.BF16).tobytes()


def f16_values(values: np.ndarray, name: str) -> np.ndarray:
    """values -> F16, nearest-even; refuses a value that is not finite or overflows F16."""
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: cannot convert non-finite weights to F16")
    with np.errstate(over="ignore"):
        h = values.astype(np.float16)
    if not np.isfinite(h).all():
        raise ValueError(f"{name}: a value is beyond F16's range, which the engine reads it as")
    return h


def convert_tensor(name: str, type_name: str, raw: np.ndarray, compat_bf16: bool):
    """The dense form of one tensor: (index kind, stored bytes, destination bytes, record or None), or an error string.
        `record` describes a conversion for meta.conversions."""
    form = form_of(name)
    if form is None or form == type_name:
        return KIND[type_name], raw.tobytes(), raw.nbytes, None
    values = _dequantize(raw, type_name)
    rec = {"src_type": type_name, "dst_type": form}
    if form == "BF16":
        u = values.view(np.uint32) if type_name == "F32" else None
        if u is not None and not np.count_nonzero(u & 0xFFFF):
            if not np.isfinite(values).all():
                return f"{name} is F32 with non-finite values; refused rather than converted"
            data = (u >> 16).astype("<u2").tobytes()       # exact BF16 values
            rec.update(method="f32->bf16: the values are BF16's (low 16 bits zero)", exact=True, max_abs_err=0.0)
            return "4", data, len(data), rec
        if not compat_bf16:
            what = "F32 with values that are not BF16" if type_name == "F32" else type_name
            return f"{name} is {what}, but the engine requires BF16; use --compat-bf16 (rounds to nearest-even)"
        data = bf16_bytes(raw, type_name)
        got = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        rec.update(method="dequantize->bf16 round-to-nearest-even (--compat-bf16)" if type_name not in FLOAT else
                   "%s->bf16 round-to-nearest-even (--compat-bf16)" % type_name.lower())
        rec.update(exact=bool(np.array_equal(got, values)),
                   max_abs_err=float(np.max(np.abs(got.astype(np.float64) - values))) if values.size else 0.0)
        return "4", data, len(data), rec
    if form == "F16":
        if type_name != "F32" and not compat_bf16:
            return f"{name} is {type_name}, but the engine reads F16; use --compat-bf16 (rounds to nearest-even)"
        h = f16_values(values, name)
        err = float(np.max(np.abs(h.astype(np.float64) - values))) if values.size else 0.0
        rec.update(exact=bool(np.array_equal(h.astype(np.float32), values)), max_abs_err=err)
        if type_name == "F32":
            # stored as F32 (index kind 3); the loader narrows to F16
            rec.update(method="f32->f16 by the loader (index kind 3, round-to-nearest-even)")
            return "3", raw.tobytes(), raw.nbytes // 2, rec
        rec.update(method="%s->f16 round-to-nearest-even (--compat-bf16)" % type_name.lower())
        data = h.astype("<f2").tobytes()
        return "5", data, len(data), rec
    # form == "F32"
    if type_name not in ("F16", "BF16") and not compat_bf16:
        return f"{name} is {type_name}, but the engine reads F32; use --compat-bf16"
    data = values.astype("<f4").tobytes()
    rec.update(method="%s->f32 widening" % type_name.lower(), exact=True, max_abs_err=0.0)
    return "2", data, len(data), rec


class Dense:
    """The `dense` section: `plan` validates every tensor first, `chunks` streams the pool and fills `rows` and `records`,
        `index_text` is the `dense.index`."""

    def __init__(self, model: SourceModel, compat_bf16: bool = False):
        self.model = model
        self.compat_bf16 = compat_bf16
        self.todo = []
        self.rows: list[list[str]] = []
        self.records: list[dict] = []
        self.pool_bytes = 0
        problems = []
        for name, (g, t, p) in model.where.items():
            if is_expert(name) or name in NOT_IN_DENSE:
                continue
            if len(t.shape) > 2:
                raise ConvertError(f"tensor {name} has {len(t.shape)} dimensions; the index holds two")
            form = form_of(name)
            # quantized tensors are served from the source unless the pool needs them dequantized (--compat-bf16)
            native = t.type_name not in FLOAT and (
                form is None or (name == "blk.1.ple_key.weight" and t.type_name in NATIVE_PLE_KEY))
            if t.type_name not in FLOAT and not native and not compat_bf16:
                problems.append(f"{name} is {t.type_name}, but the engine requires {form}; use --compat-bf16")
            self.todo.append((name, t, p, native))
        if problems:
            more = f"\n... and {len(problems) - 8} more" if len(problems) > 8 else ""
            raise ConvertError("\n".join(problems[:8]) + more)

    def chunks(self):
        at = 0
        for name, t, p, native in self.todo:
            ne0 = int(t.shape[0])
            ne1 = int(t.shape[1]) if len(t.shape) > 1 else 0
            if native:
                self.rows.append([name, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1), "8", "0", "32"] + ["0"] * 7)
                continue
            raw = self.model.bytes(name)
            try:
                got = convert_tensor(name, t.type_name, raw, self.compat_bf16)
            except ValueError as e:
                got = str(e)
            if isinstance(got, str):
                raise ConvertError(got)
            kind, data, dst_bytes, rec = got
            if rec is not None:
                self.records.append({"name": name, "shard": p.name, **rec,
                                     "source_sha256": hashlib.sha256(raw.tobytes()).hexdigest()})
            self.rows.append([name, "0", kind, str(at), str(len(data)), "0", str(dst_bytes), str(ne0), str(ne1),
                              "0", "0", "1"] + ["0"] * 7)
            yield data
            pad = (-len(data)) % ALIGN
            if pad:
                yield b"\0" * pad
            at += len(data) + pad

    def index_text(self) -> bytes:
        at = 0
        for r in self.rows:
            r[5] = str(at)
            at += (int(r[6]) + ALIGN - 1) // ALIGN * ALIGN
        self.pool_bytes = at
        out = ["# dense index v3\n", "# align %d pool %d tensors %d\n" % (ALIGN, at, len(self.rows))]
        out += [" ".join(r) + "\n" for r in self.rows]
        return "".join(out).encode("utf-8")


# ------------------------------------------------------------------ experts
class Experts:
    """The `experts` section and its `experts.layout`; each role is resolved by name in the shard that holds it."""

    def __init__(self, model: SourceModel):
        T = {n: w[1] for n, w in model.where.items()}
        exps = [n for n in T if n.startswith("blk.") and n.endswith("_exps.weight")]
        if not exps:
            raise ConvertError("the model has no expert tensors (blk.N.ffn_{gate,up,down}_exps.weight)")
        self.model = model
        self.n_layers = 1 + max(int(n.split(".")[1]) for n in exps)
        # router rows = experts kept (a pruned model has fewer)
        self.n_expert = int(T["blk.0.ffn_gate_inp.weight"].shape[1])
        if any(int(T["blk.%d.ffn_gate_inp.weight" % l].shape[1]) != self.n_expert for l in range(self.n_layers)):
            raise ConvertError("the routers disagree on the expert count; a per-layer pruned model cannot be converted")
        self.layout = []
        offset = 0
        for l in range(self.n_layers):
            names = ["blk.%d.ffn_%s_exps.weight" % (l, r) for r in ROLES]
            if any(n not in T for n in names):
                raise ConvertError("layer %d: missing %s" % (l, ", ".join(n for n in names if n not in T)))
            ts = [T[n] for n in names]
            if any(t.expected_bytes() is None or len(t.shape) != 3 or int(t.shape[2]) != self.n_expert for t in ts):
                raise ConvertError("layer %d: an expert tensor is not [*, *, %d] of whole blocks" % (l, self.n_expert))
            per = [t.expected_bytes() // self.n_expert for t in ts]
            if per[0] != per[1] or ts[0].type_name != ts[1].type_name:
                raise ConvertError("layer %d: gate and up differ in type" % l)
            blob = sum(per)
            self.layout.append((l, ts[0].type_id, ts[2].type_id, offset, blob, names))
            offset += blob * self.n_expert
        self.total = offset

    def layout_text(self) -> bytes:
        head = ("# experts v5: layer gu_type d_type offset blob_bytes (n_expert %d, total %d)\n"
                % (self.n_expert, self.total))
        return (head + "".join("%d %d %d %d %d\n" % (l, gt, dt, off, blob)
                               for l, gt, dt, off, blob, _ in self.layout)).encode("utf-8")

    def chunks(self, group: int = 32):
        """The blobs layer by layer, `group` experts at a time: each expert is [gate | up | down] of its slices."""
        for l, gt, dt, off, blob, names in self.layout:
            arrays = [self.model.bytes(n).reshape(self.n_expert, -1) for n in names]
            for s in range(0, self.n_expert, group):
                yield np.concatenate([a[s:s + group] for a in arrays], axis=1).tobytes()


# ------------------------------------------------------------------ the drafter
MTP_H, MTP_FF, MTP_NE = 2560, 640, 512
MTP_EXPERTS = ("mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj")
MTP_Q8 = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
          "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.index_qk_proj.weight",
          "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
          "mlp.shared_expert.down_proj.weight"}


def q2_0(x: np.ndarray) -> np.ndarray:
    """Float32 values in blocks of 64 -> Q2_0 blocks (fp16 d, 16 bytes of 2-bit codes, code = q + 1, four per byte).
        The block scale minimizes squared error over {-1, 0, 1, 2} x d (the ggml reference uses d = max|w| and never the +2 level)."""
    x = x.reshape(-1, 64).astype(np.float32)
    amax = np.abs(x).max(axis=1, keepdims=True)
    best_err = np.full((x.shape[0], 1), np.inf, dtype=np.float32)
    best_d = np.zeros_like(amax)
    for f in np.linspace(0.5, 1.0, 17, dtype=np.float32):      # scale search range: amax/2 .. amax
        d = amax * f
        inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        q = np.clip(np.rint(x * inv), -1, 2)
        err = ((q * d - x) ** 2).sum(axis=1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_d = np.where(better, d, best_d)
    d16 = best_d.astype(np.float16)
    d = d16.astype(np.float32)                                   # quantize against the stored (fp16) scale
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    codes = (np.clip(np.rint(x * inv), -1, 2) + 1).astype(np.uint8)
    c = codes.reshape(-1, 16, 4)
    packed = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    out = np.empty((x.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = packed
    return out.reshape(-1)


def q8_0(x: np.ndarray) -> bytes:
    """ggml's reference rounding over rows of a float32 array: blocks of 32, fp16 d = amax / 127."""
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(b).max(axis=1) / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def mtp_blob(gu: np.ndarray, dn: np.ndarray) -> bytes:
    """One expert in the engine's blob layout: gu is its (1280, 40, 18) Q2_0 rows (gate 0..639, up 640..1279), dn its
        (2560, 10, 18). Gate/up rows interleave (2r = gate r, 2r+1 = up r), then the down rows; codes of all rows in one
        plane, fp16 scales in another."""
    inter = np.empty_like(gu)
    inter[0::2] = gu[:MTP_FF]
    inter[1::2] = gu[MTP_FF:]
    out = b"".join([inter[:, :, 2:].tobytes(), dn[:, :, 2:].tobytes(), inter[:, :, :2].tobytes(),
                    dn[:, :, :2].tobytes()])
    assert len(out) == 3 * (MTP_H * MTP_FF * 18 // 64), len(out)
    return out


class Mtp:
    """The draft layer's sections, from a directory written by `mtp_fetch.py fetch`."""

    def __init__(self, src):
        self.src = pathlib.Path(src)
        mf = self.src / "mtp-manifest.json"
        if not mf.is_file():
            raise ConvertError(f"{mf} not found (run tools/convert/mtp_fetch.py fetch --out {self.src})")
        self.manifest = {t["name"]: t for t in json.loads(mf.read_text())}
        for n in MTP_EXPERTS:
            if n not in self.manifest:
                raise ConvertError(f"{mf}: tensor {n} is missing")
        self.dense_lines: list[str] = []
        self.dense_bytes = 0

    def source_sha256(self) -> str:
        return hashlib.sha256(json.dumps({n: t["sha256"] for n, t in self.manifest.items()},
                                         sort_keys=True).encode()).hexdigest()

    def dense_chunks(self):
        """Every tensor but the experts, by name, each starting at a multiple of 256; fills `dense_lines`."""
        off = 0
        for name in sorted(self.manifest):
            if name in MTP_EXPERTS:
                continue
            t = self.manifest[name]
            shape = t["shape"]
            short = name[len("mtp."):] if name.startswith("mtp.") else name
            short = short[len("layers.0."):] if short.startswith("layers.0.") else short
            raw16 = np.fromfile(self.src / t["file"], dtype="<u2")
            if raw16.size != int(np.prod(shape)):
                raise ConvertError(f"{t['file']}: {raw16.size} values, expected shape {shape}")
            if len(shape) == 1:      # norm weights: apply the Gemma RMSNorm "+1"
                arr = _bf16_to_f32(raw16) + 1.0
                raw, kind, rows, cols = arr.astype(np.float32).tobytes(), "f32", 1, arr.size
            else:
                u16 = raw16.reshape(shape[0], -1)
                rows, cols = u16.shape
                if short in MTP_Q8:
                    raw, kind = q8_0(_bf16_to_f32(u16)), "q8_0"
                else:
                    raw, kind = u16.tobytes(), "bf16"
            pad = (-off) % 256
            if pad:
                yield b"\0" * pad
                off += pad
            yield raw
            self.dense_lines.append(f"{short} {kind} {rows} {cols} {off} {len(raw)}")
            off += len(raw)
        self.dense_bytes = off

    def dense_index(self) -> bytes:
        return ("\n".join(self.dense_lines) + "\n").encode("utf-8")

    def expert_chunks(self):
        """512 blobs, one per expert; each expert is quantized to Q2_0 on its own, so the memory stays small."""
        gu_t, dn_t = (self.manifest[n] for n in MTP_EXPERTS)
        if gu_t["shape"] != [MTP_NE, 2 * MTP_FF, MTP_H] or dn_t["shape"] != [MTP_NE, MTP_H, MTP_FF]:
            raise ConvertError(f"unexpected drafter expert shapes {gu_t['shape']} {dn_t['shape']}")
        gu = np.memmap(self.src / gu_t["file"], dtype="<u2", mode="r", shape=tuple(gu_t["shape"]))
        dn = np.memmap(self.src / dn_t["file"], dtype="<u2", mode="r", shape=tuple(dn_t["shape"]))
        for e in range(MTP_NE):
            g = q2_0(_bf16_to_f32(np.asarray(gu[e]))).reshape(2 * MTP_FF, MTP_H // 64, 18)
            d = q2_0(_bf16_to_f32(np.asarray(dn[e]))).reshape(MTP_H, MTP_FF // 64, 18)
            yield mtp_blob(g, d)


def draft_vocab_bytes(path, n_vocab: int | None = None) -> bytes:
    """A draft vocabulary file: int32 token ids, none outside the vocabulary."""
    data = pathlib.Path(path).read_bytes()
    if len(data) % 4:
        raise ConvertError(f"{path}: not a list of int32 ids ({len(data)} bytes)")
    ids = np.frombuffer(data, dtype="<i4")
    if ids.size == 0 or (ids < 0).any() or (n_vocab is not None and (ids >= n_vocab).any()):
        raise ConvertError(f"{path}: an id lies outside the vocabulary" if ids.size else f"{path}: empty")
    return data


# ------------------------------------------------------------------ the expert profile
PROFILE_HEADER = struct.Struct("<5I")


def read_trace(path, n_layers: int, n_expert: int) -> dict:
    """(layer, expert) -> count over `--dump-routing` records (int32 layer, int32 k, k int32 ids, k float32 weights)."""
    blob = pathlib.Path(path).read_bytes()
    off, freq = 0, defaultdict(int)
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        if k < 0 or off + 8 * k > len(blob):
            raise ConvertError(f"{path}: a truncated routing record")
        for e in struct.unpack_from("<%di" % k, blob, off):
            if 0 <= layer < n_layers and 0 <= e < n_expert:
                freq[(layer, e)] += 1
        off += 8 * k
    return freq


def rank_pairs(freq: dict, n_layers: int, n_expert: int) -> list:
    """Routed pairs, most frequent first, then every other pair interleaved across layers (expert-major)."""
    ranked, seen = [], set()
    for p, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
        seen.add(p)
        ranked.append(p)
    for e in range(n_expert):
        for layer in range(n_layers):
            if (layer, e) not in seen:
                ranked.append((layer, e))
    return ranked


def build_profile(ranked: list, n_layers: int, n_expert: int) -> bytes:
    """The profile section: version, n_layers, n_expert, slots, n_ranked (uint32), the pairs (uint16 layer, uint16
        expert), then the n_layers x n_expert int32 rank table (-1: not ranked)."""
    table = [[-1] * n_expert for _ in range(n_layers)]
    for slot, (layer, e) in enumerate(ranked):
        table[layer][e] = slot
    out = [PROFILE_HEADER.pack(1, n_layers, n_expert, len(ranked), len(ranked))]
    out += [struct.pack("<HH", layer, e) for layer, e in ranked]
    out += [struct.pack("<%di" % n_expert, *row) for row in table]
    return b"".join(out)


def profile_info(blob: bytes, n_layers: int | None = None, n_expert: int | None = None) -> dict:
    """Checks a profile section and describes it."""
    if len(blob) < PROFILE_HEADER.size:
        raise ConvertError("profile: too short to hold a header")
    ver, nl, ne, slots, n = PROFILE_HEADER.unpack_from(blob, 0)
    if (n_layers is not None and nl != n_layers) or (n_expert is not None and ne != n_expert):
        raise ConvertError(f"profile: {nl}x{ne}, but the model is {n_layers}x{n_expert}")
    want = PROFILE_HEADER.size + 4 * n + 4 * nl * ne
    if len(blob) != want:
        raise ConvertError(f"profile: {len(blob)} bytes, expected {want} for {n} pairs of {nl}x{ne}")
    return {"version": ver, "n_layer": nl, "n_expert": ne, "slots": slots, "n_pairs": n,
            "layout": "5 x u32 LE (version, n_layer, n_expert, slots, n_pairs); then pairs: n_pairs x (u16 layer, "
                      "u16 expert) in rank order; then table: n_layer x n_expert x i32 LE, the rank of each "
                      "(layer, expert) or -1",
            "pairs_offset": PROFILE_HEADER.size, "table_offset": PROFILE_HEADER.size + 4 * n}
