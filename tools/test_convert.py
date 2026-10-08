"""Tests for tools/convert.py and tools/model_sections.py on a tiny synthetic model, drafter and side files.

    python3 -m unittest discover -s tools -p 'test_convert.py'
"""
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert  # noqa: E402
import model_sections as S  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import ModelFile  # noqa: E402

N_EXPERT, N_LAYERS = 4, 2
TOKENS = ["a", "b", "c", "d", "e", "f"]


def small_ints(rng, shape):
    """Float32 values that are exactly BF16."""
    return rng.integers(-8, 9, size=shape).astype(np.float32)


class ConvertTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.t = t
        rng = np.random.default_rng(0)
        g = GGUFWriter()
        g.add("general.architecture", "qwen")
        g.add("general.quantized_by", "Someone")
        g.add("qwen.block_count", N_LAYERS, "u32")
        g.add("qwen.rope.freqs", [0.5, 1.5], "array:f32")
        g.add("qwen.nan", float("inf"), "f32")
        g.add("tokenizer.ggml.tokens", TOKENS)
        self.router = small_ints(rng, (8, N_EXPERT))
        self.attn = rng.standard_normal((3, 8)).astype(np.float32)
        self.embd = rng.standard_normal((5, 8)).astype(np.float32)
        self.conv = rng.standard_normal((2, 4)).astype(np.float32)
        self.ple = rng.standard_normal((2, 6)).astype(np.float32)
        self.q2 = rng.standard_normal((2, 64)).astype(np.float32)
        self.experts = {}
        g.add_f32("token_embd.weight", self.embd)
        g.add_f32("per_layer_token_embd.weight", self.ple)
        for l in range(N_LAYERS):
            g.add_f32(f"blk.{l}.ffn_gate_inp.weight", self.router)
            g.add_f32(f"blk.{l}.attn_q.weight", self.attn)
            for role, shape in (("gate", (3, 2, N_EXPERT)), ("up", (3, 2, N_EXPERT)), ("down", (2, 3, N_EXPERT))):
                a = rng.standard_normal(shape).astype(np.float32)
                self.experts[(l, role)] = a
                g.add_f32(f"blk.{l}.ffn_{role}_exps.weight", a)
        g.add_f32("blk.0.ple_conv1d.weight", self.conv)
        g.add_q2_0("blk.0.attn_gate.weight", self.q2, [64, 2])
        self.gguf = g.write(t / "m-00001-of-00001.gguf")
        self.tok = {"tokenizer.json": b"{}", "vocab.json": b'{"a":0}', "merges.txt": b"a b\n",
                    "token_type.json": b"[1]", "chat_template.jinja": b"{{ x }}"}
        self.hc8 = rng.bytes(123)
        (t / "hc8.bin").write_bytes(self.hc8)
        self.mmproj = rng.bytes(4321)
        (t / "mmproj.gguf").write_bytes(self.mmproj)
        self.vision = None
        self._orig = (convert.load_vision_helper, convert.export_tokenizer)
        convert.load_vision_helper = lambda: self.vision

        def tokenizer_files(first, out_dir):
            d = Path(out_dir) / "tokenizer"
            d.mkdir(parents=True, exist_ok=True)
            for k, v in self.tok.items():
                (d / k).write_bytes(v)
            return sorted(d.iterdir())
        convert.export_tokenizer = tokenizer_files

    def tearDown(self):
        convert.load_vision_helper, convert.export_tokenizer = self._orig
        self.tmp.cleanup()

    def args(self, **kw):
        t = self.t
        d = dict(gguf=[str(self.gguf)], mmproj=str(t / "mmproj.gguf"), out=str(t / "out.basalt"), mtp_source=None,
                 draft_vocab=None, profile=None, profile_from=None, hc8=str(t / "hc8.bin"), name="Tiny",
                 compat_bf16=False, no_vision=False, force=False)
        d.update(kw)
        return type("A", (), d)()

    def model(self):
        return ModelFile(str(self.t / "out.basalt"))

    # ---------------------------------------------------------------- sections
    def test_sections(self):
        convert.convert(self.args())
        m = self.model()
        self.assertFalse((self.t / "out.basalt.partial").exists())
        want = {"hc8", "vision/mmproj", "profile", "dense", "dense.index", "experts", "experts.layout", "gguf", "meta"}
        want |= {"tokenizer/" + k for k in self.tok}
        self.assertEqual(set(m.names()), want)
        for k in m.names():
            self.assertTrue(m.verify(k), k)
        self.assertEqual(m.read("hc8"), self.hc8)
        self.assertEqual(m.read("vision/mmproj"), self.mmproj)
        for k, v in self.tok.items():
            self.assertEqual(m.read("tokenizer/" + k), v)

    def test_experts_and_layout(self):
        convert.convert(self.args())
        m = self.model()
        blobs = b""
        for l in range(N_LAYERS):
            parts = [self.experts[(l, r)].reshape(N_EXPERT, -1) for r in ("gate", "up", "down")]
            blobs += np.concatenate(parts, axis=1).tobytes()
        self.assertEqual(m.read("experts"), blobs)
        blob = (24 * 3) // N_EXPERT * 4              # bytes per expert: gate + up + down, F32
        lines = m.read("experts.layout").decode().splitlines()
        self.assertEqual(lines[0], f"# experts v5: layer gu_type d_type offset blob_bytes "
                                   f"(n_expert {N_EXPERT}, total {2 * blob * N_EXPERT})")
        self.assertEqual(lines[1:], [f"0 0 0 0 {blob}", f"1 0 0 {blob * N_EXPERT} {blob}"])

    def test_dense(self):
        convert.convert(self.args())
        m = self.model()
        rows = [l.split() for l in m.read("dense.index").decode().splitlines()]
        self.assertEqual(rows[0], ["#", "dense", "index", "v3"])
        by = {r[0]: r for r in rows[2:]}
        pool = m.read("dense")
        # the router is BF16 in the pool (exact narrowing); a plain float tensor is stored as it is
        r = by["blk.0.ffn_gate_inp.weight"]
        self.assertEqual(r[2], "4")
        off, n = int(r[3]), int(r[4])
        self.assertEqual(pool[off:off + n], (self.router.view(np.uint32) >> 16).astype("<u2").tobytes())
        r = by["blk.0.attn_q.weight"]
        self.assertEqual(r[2], "2")
        self.assertEqual(pool[int(r[3]):int(r[3]) + int(r[4])], self.attn.astype("<f4").tobytes())
        # F32 PLE conv: kind 3 (narrowed by the loader), 2 bytes an element in the arena
        r = by["blk.0.ple_conv1d.weight"]
        self.assertEqual((r[2], int(r[4]), int(r[6])), ("3", self.conv.nbytes, self.conv.nbytes // 2))
        # a quantized tensor is only listed; the PLE table and the experts are not listed at all
        self.assertEqual(by["blk.0.attn_gate.weight"][4], "0")
        self.assertNotIn("per_layer_token_embd.weight", by)
        self.assertFalse(any("_exps" in k for k in by))
        self.assertEqual(rows[1][:2], ["#", "align"])
        arena = [int(r[5]) for r in rows[2:]]
        self.assertEqual(arena, sorted(arena))
        self.assertTrue(all(a % 64 == 0 for a in arena))
        self.assertEqual(int(rows[1][4]), (arena[-1] + (int(rows[-1][6]) + 63) // 64 * 64))

    def test_gguf_section_and_meta(self):
        convert.convert(self.args())
        m = self.model()
        meta = json.loads(m.read("meta"))
        self.assertEqual(meta["format"], "basalt")
        self.assertEqual(meta["name"], "Tiny")
        self.assertEqual(meta["gguf_kv"]["qwen.block_count"], N_LAYERS)
        self.assertEqual(meta["gguf_kv"]["qwen.rope.freqs"], [0.5, 1.5])
        self.assertEqual(meta["gguf_kv"]["qwen.nan"], "inf")
        tens = meta["tensors"]
        self.assertFalse(any("_exps" in k for k in tens))
        self.assertIn("per_layer_token_embd.weight", tens)
        self.assertEqual(tens["token_embd.weight"]["offset"], 0)
        gg = m.read("gguf")
        for name, arr in (("token_embd.weight", self.embd), ("per_layer_token_embd.weight", self.ple)):
            o = tens[name]["offset"]
            self.assertEqual(gg[o:o + arr.nbytes], arr.astype("<f4").tobytes(), name)
            self.assertEqual(o % 4096, 0)
        self.assertEqual(tens["blk.0.attn_gate.weight"]["type"], "Q2_0")
        self.assertEqual(meta["hc8"], {"bytes": 123})
        self.assertEqual(meta["vision"]["mmproj"], "vision/mmproj")
        self.assertEqual(meta["sources"][0]["size"], self.gguf.stat().st_size)
        self.assertEqual(meta["profile"]["n_pairs"], N_LAYERS * N_EXPERT)
        self.assertNotIn("engine_args", meta)
        self.assertEqual(meta["sampling"]["top_k"], 20)
        self.assertIsNone(meta["mtp"])
        lic = meta["license"]
        self.assertEqual(lic["name"], "Qwen Community License 1.0")
        self.assertEqual(lic["text"], (Path(convert.__file__).parent / "qwen-community-license-1.0.txt").read_text())
        self.assertEqual(lic["quantized_by"], "Someone")
        self.assertEqual(set(lic), {"name", "text", "quantized_by"})

    def test_without_hc8_or_drafter(self):
        convert.convert(self.args(hc8=None))
        m = self.model()
        self.assertNotIn("hc8", m.names())
        self.assertFalse([n for n in m.names() if n.startswith("mtp/")])
        self.assertIsNone(json.loads(m.read("meta"))["hc8"])

    # ---------------------------------------------------------------- profile
    def test_profile_fill_order_and_traces(self):
        convert.convert(self.args())
        blob = self.model().read("profile")
        info = S.profile_info(blob, N_LAYERS, N_EXPERT)
        self.assertEqual(info["slots"], 8)
        pairs = struct.unpack_from("<16H", blob, 20)
        self.assertEqual(pairs[:4], (0, 0, 1, 0))                    # expert-major fill: (0,0) (1,0) (0,1) ...
        # a trace of three records (each routes its expert and expert 0): (1,3) and (1,0) twice, (0,2) and (0,0) once
        trace = b"".join(struct.pack("<ii2i2f", l, 2, e, 0, 1.0, 1.0) for l, e in ((1, 3), (0, 2), (1, 3)))
        (self.t / "trace.bin").write_bytes(trace)
        convert.convert(self.args(profile_from=[str(self.t / "trace.bin")], force=True))
        blob = self.model().read("profile")
        pairs = struct.unpack_from("<16H", blob, 20)
        self.assertEqual(pairs[:8], (1, 0, 1, 3, 0, 0, 0, 2))   # by count, then by pair
        table = struct.unpack_from("<8i", blob, 20 + 4 * 8)
        self.assertEqual(table[1 * N_EXPERT + 3], 1)

    def test_profile_file(self):
        ranked = [(l, e) for e in range(N_EXPERT) for l in range(N_LAYERS)][::-1]
        (self.t / "p.bin").write_bytes(S.build_profile(ranked, N_LAYERS, N_EXPERT))
        convert.convert(self.args(profile=str(self.t / "p.bin")))
        self.assertEqual(self.model().read("profile"), (self.t / "p.bin").read_bytes())

    def test_bad_profile(self):
        good = S.build_profile([(0, 0)], N_LAYERS, N_EXPERT)
        for name, blob in (("short", good[:-4]), ("other model", S.build_profile([(0, 0)], N_LAYERS, N_EXPERT + 1))):
            (self.t / "p.bin").write_bytes(blob)
            with self.assertRaises(SystemExit, msg=name):
                convert.convert(self.args(profile=str(self.t / "p.bin")))
            self.assertFalse((self.t / "out.basalt").exists())

    # ---------------------------------------------------------------- refusals and options
    def test_no_vision_and_overwrite(self):
        convert.convert(self.args(no_vision=True))
        self.assertNotIn("vision/mmproj", self.model().names())
        with self.assertRaises(SystemExit):
            convert.convert(self.args())
        convert.convert(self.args(force=True))
        self.assertIn("vision/mmproj", self.model().names())

    def test_vision_sections(self):
        src = self.t / "extra.bin"
        src.write_bytes(b"0123456789")
        self.vision = lambda mm, nat: [("vision/a", b"inline", None, 0, 6), ("vision/b", None, str(src), 2, 5)]
        convert.convert(self.args())
        m = self.model()
        self.assertEqual(m.read("vision/a"), b"inline")
        self.assertEqual(m.read("vision/b"), b"23456")
        self.assertEqual(json.loads(m.read("meta"))["vision"]["extra"], ["vision/a", "vision/b"])

    def test_router_not_bf16_is_refused(self):
        g = GGUFWriter()
        g.add("general.architecture", "qwen")
        g.add_f32("blk.0.ffn_gate_inp.weight", np.full((8, N_EXPERT), 0.10001, dtype=np.float32))
        for role, shape in (("gate", (3, 2, N_EXPERT)), ("up", (3, 2, N_EXPERT)), ("down", (2, 3, N_EXPERT))):
            g.add_f32(f"blk.0.ffn_{role}_exps.weight", np.zeros(shape, dtype=np.float32))
        path = g.write(self.t / "bad.gguf")
        with self.assertRaises(SystemExit) as cm:
            convert.convert(self.args(gguf=[str(path)]))
        self.assertIn("requires BF16", str(cm.exception))
        self.assertFalse((self.t / "out.basalt").exists())
        self.assertFalse((self.t / "out.basalt.partial").exists())

    def test_draft_vocab_needs_the_drafter(self):
        (self.t / "dv.bin").write_bytes(np.array([0, 1], dtype="<i4").tobytes())
        with self.assertRaises(SystemExit):
            convert.convert(self.args(draft_vocab=str(self.t / "dv.bin")))

    def test_help(self):
        r = subprocess.run([sys.executable, str(Path(convert.__file__)), "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        for flag in ("--gguf", "--mtp-source", "--mmproj", "--profile-from", "--profile ", "--hc8", "--draft-vocab"):
            self.assertIn(flag, r.stdout)


class DrafterTest(unittest.TestCase):
    """The draft layer's sections at tiny dimensions."""

    H, FF, NE = 64, 64, 2

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name)
        self._dims = (S.MTP_H, S.MTP_FF, S.MTP_NE)
        S.MTP_H, S.MTP_FF, S.MTP_NE = self.H, self.FF, self.NE
        rng = np.random.default_rng(1)
        manifest, self.vals = [], {}

        def put(name, shape):
            u16 = (rng.standard_normal(shape).astype(np.float32).view(np.uint32) >> 16).astype("<u2")
            f = name.replace(".", "_") + ".bin"
            u16.tofile(self.src / f)
            self.vals[name] = (u16.astype(np.uint32) << 16).view(np.float32)
            manifest.append({"name": name, "shape": list(shape), "file": f, "sha256": "0" * 64})

        put("mtp.fc_embedding.weight", (4, 64))                       # Q8_0
        put("mtp.hyper_connection_mixer.hc_norm.weight", (8,))        # norm: F32, +1
        put("mtp.layers.0.mlp.gate.weight", (4, 64))                  # BF16 as it is
        put("mtp.layers.0.mlp.experts.gate_up_proj", (self.NE, 2 * self.FF, self.H))
        put("mtp.layers.0.mlp.experts.down_proj", (self.NE, self.H, self.FF))
        (self.src / "mtp-manifest.json").write_text(json.dumps(manifest))

    def tearDown(self):
        S.MTP_H, S.MTP_FF, S.MTP_NE = self._dims
        self.tmp.cleanup()

    def test_dense(self):
        m = S.Mtp(self.src)
        blob = b"".join(m.dense_chunks())
        lines = [l.split() for l in m.dense_index().decode().splitlines()]
        self.assertEqual([l[0] for l in lines],
                         ["fc_embedding.weight", "hyper_connection_mixer.hc_norm.weight", "mlp.gate.weight"])
        by = {l[0]: l for l in lines}
        kind, rows, cols, off, n = by["fc_embedding.weight"][1:]
        self.assertEqual((kind, rows, cols, off), ("q8_0", "4", "64", "0"))
        self.assertEqual(blob[:int(n)], S.q8_0(self.vals["mtp.fc_embedding.weight"]))
        kind, rows, cols, off, n = by["hyper_connection_mixer.hc_norm.weight"][1:]
        self.assertEqual((kind, rows, cols), ("f32", "1", "8"))
        self.assertEqual(int(off) % 256, 0)
        got = np.frombuffer(blob[int(off):int(off) + int(n)], dtype="<f4")
        np.testing.assert_array_equal(got, self.vals["mtp.hyper_connection_mixer.hc_norm.weight"] + 1.0)
        kind, rows, cols, off, n = by["mlp.gate.weight"][1:]
        self.assertEqual(kind, "bf16")
        got = np.frombuffer(blob[int(off):int(off) + int(n)], dtype="<u2")
        np.testing.assert_array_equal((got.astype(np.uint32) << 16).view(np.float32).reshape(4, 64),
                                      self.vals["mtp.layers.0.mlp.gate.weight"])

    def test_experts(self):
        m = S.Mtp(self.src)
        blobs = list(m.expert_chunks())
        self.assertEqual(len(blobs), self.NE)
        size = 3 * (self.H * self.FF * 18 // 64)
        self.assertTrue(all(len(b) == size for b in blobs))
        # layout: the gate/up code plane first (rows interleaved: gate r, up r), then the down codes, then the scales
        gu = self.vals["mtp.layers.0.mlp.experts.gate_up_proj"][1]
        q_gate0 = S.q2_0(gu[0]).reshape(1, 18)
        q_up0 = S.q2_0(gu[self.FF]).reshape(1, 18)
        codes = np.frombuffer(blobs[1], dtype=np.uint8)
        self.assertEqual(codes[0:16].tobytes(), q_gate0[0, 2:].tobytes())
        self.assertEqual(codes[16:32].tobytes(), q_up0[0, 2:].tobytes())
        scales_at = 2 * self.FF * 16 + self.H * (self.FF // 64) * 16
        self.assertEqual(blobs[1][scales_at:scales_at + 2], q_gate0[0, :2].tobytes())
        self.assertEqual(blobs[1][scales_at + 2:scales_at + 4], q_up0[0, :2].tobytes())

    def test_q2_0_is_a_valid_quantization(self):
        x = np.random.default_rng(2).standard_normal(256).astype(np.float32)
        b = S.q2_0(x).reshape(-1, 18)
        d = b[:, :2].copy().view(np.float16).astype(np.float32)
        qs = b[:, 2:]
        codes = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(-1, 64).astype(np.float32)
        back = ((codes - 1.0) * d).reshape(-1)
        self.assertLess(float(np.abs(back - x).mean()), 0.5 * float(np.abs(x).mean()))

    def test_in_a_model(self):
        t = ConvertTest("test_sections")
        t.setUp()
        try:
            (t.t / "dv.bin").write_bytes(np.array([0, 1, 5], dtype="<i4").tobytes())
            convert.convert(t.args(mtp_source=str(self.src), draft_vocab=str(t.t / "dv.bin")))
            m = t.model()
            self.assertEqual({n for n in m.names() if n.startswith("mtp/")},
                             {"mtp/dense", "mtp/dense.index", "mtp/experts", "mtp/draft_vocab"})
            self.assertEqual(len(m.read("mtp/experts")), self.NE * 3 * (self.H * self.FF * 18 // 64))
            self.assertEqual(m.read("mtp/draft_vocab"), (t.t / "dv.bin").read_bytes())
            (t.t / "dv.bin").write_bytes(np.array([0, 6], dtype="<i4").tobytes())      # 6 is outside the 6 tokens
            with self.assertRaises(SystemExit):
                convert.convert(t.args(mtp_source=str(self.src), draft_vocab=str(t.t / "dv.bin"), force=True))
        finally:
            t.tearDown()


if __name__ == "__main__":
    unittest.main()
