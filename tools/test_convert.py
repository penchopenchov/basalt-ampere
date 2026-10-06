"""Tests for tools/convert.py on a tiny synthetic pack, GGUF, mtp directory and side files."""
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import ModelFile  # noqa: E402

INDEX = b"# strata pack index v3 -- tensors\nblk.0.attn_q 0 16\nblk.0.attn_k 16 16\n"
LAYOUT = b"# strata native experts v3: layer gu_type d_type offset blob_bytes (n_expert 4, total 64)\n0 18 18 0 16\n"
MTP_INDEX = b"fc_embedding.weight q8_0 4 4 0 16\n"


def make_profile(nl=2, ne=3):
    pairs = [(l, e) for e in range(ne) for l in range(nl)]
    blob = b"STRP" + struct.pack("<5I", 1, nl, ne, len(pairs), len(pairs))
    blob += b"".join(struct.pack("<HH", *p) for p in pairs)
    for l in range(nl):
        blob += struct.pack("<%di" % ne, *[pairs.index((l, e)) for e in range(ne)])
    return blob


class ConvertTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.t = t
        rng = np.random.default_rng(0)
        pack = t / "pack"
        (pack / "tokenizer").mkdir(parents=True)
        self.dense = rng.bytes(5000)
        self.experts = rng.bytes(3000)
        (pack / "dense.bin").write_bytes(self.dense)
        (pack / "index.txt").write_bytes(INDEX)
        (pack / "experts.bin").write_bytes(self.experts)
        (pack / "native_experts.txt").write_bytes(LAYOUT)
        self.tok = {"tokenizer.json": b"{}", "vocab.json": b'{"a":0}', "merges.txt": b"a b\n",
                    "token_type.json": b"[1]", "chat_template.jinja": b"{{ x }}"}
        for k, v in self.tok.items():
            (pack / "tokenizer" / k).write_bytes(v)
        g = GGUFWriter()
        g.add("general.architecture", "qwen")
        g.add("qwen.block_count", 2, "u32")
        g.add("qwen.rope.freqs", [0.5, 1.5], "array:f32")
        g.add("qwen.nan", float("inf"), "f32")
        self.a = rng.standard_normal((3, 8)).astype(np.float32)
        self.b = rng.standard_normal((2, 4)).astype(np.float32)
        g.add_f32("token_embd.weight", self.a)
        g.add_f32("blk.0.ffn_gate_exps.weight", rng.standard_normal((4, 64)).astype(np.float32))
        g.add_f32("output_norm.weight", self.b)
        self.gguf = g.write(t / "m-00001-of-00001.gguf")
        mtp = t / "mtp"
        mtp.mkdir()
        self.mtp = {"dense.bin": rng.bytes(700), "dense.txt": MTP_INDEX, "experts.bin": rng.bytes(900),
                    "draft_vocab.bin": rng.bytes(64)}
        for k, v in self.mtp.items():
            (mtp / k).write_bytes(v)
        self.profile = make_profile()
        (t / "profile.bin").write_bytes(self.profile)
        self.hc8 = rng.bytes(123)
        (t / "hc8.bin").write_bytes(self.hc8)
        self.mmproj = rng.bytes(4321)
        (t / "mmproj.gguf").write_bytes(self.mmproj)

        self.vision = None
        self._orig = convert.load_vision_helper
        convert.load_vision_helper = lambda: self.vision

    def tearDown(self):
        convert.load_vision_helper = self._orig
        self.tmp.cleanup()

    def args(self, **kw):
        t = self.t
        d = dict(pack=str(t / "pack"), native=[str(self.gguf)], mtp=str(t / "mtp"), profile=str(t / "profile.bin"),
                 hc8=str(t / "hc8.bin"), mmproj=str(t / "mmproj.gguf"), out=str(t / "out.basalt"),
                 name="Tiny", no_vision=False, force=False)
        d.update(kw)
        return type("A", (), d)()

    def test_sections(self):
        convert.convert(self.args())
        m = ModelFile(str(self.t / "out.basalt"))
        self.assertFalse((self.t / "out.basalt.partial").exists())
        a, b_ = self.a.astype("<f4").tobytes(), self.b.astype("<f4").tobytes()
        want = {"dense": self.dense, "experts": self.experts, "hc8": self.hc8, "vision/mmproj": self.mmproj,
                "profile": self.profile[4:], "mtp/dense": self.mtp["dense.bin"],
                "mtp/experts": self.mtp["experts.bin"], "mtp/draft_vocab": self.mtp["draft_vocab.bin"],
                "mtp/dense.index": MTP_INDEX,
                "dense.index": b"# dense index v3\nblk.0.attn_q 0 16\nblk.0.attn_k 16 16\n",
                "experts.layout": LAYOUT.replace(b"# strata native experts v3", b"# experts v3"),
                "gguf": a + b"\0" * (4096 - len(a)) + b_}
        for k, v in self.tok.items():
            want["tokenizer/" + k] = v
        self.assertEqual(set(m.names()), set(want) | {"meta"})
        for k, v in want.items():
            self.assertEqual(m.read(k), v, k)
        for k in m.names():
            self.assertTrue(m.verify(k), k)
        self.assertNotIn(b"ffn_gate_exps", m.read("meta").split(b'"tensors"')[1].split(b'"engine_args"')[0])
        for k in ("dense.index", "experts.layout", "mtp/dense.index", "profile"):
            self.assertNotIn(b"strata", m.read(k).lower(), k)
        meta = json.loads(m.read("meta"))
        meta.pop("license")                 # the attribution names the upstream project on purpose
        self.assertNotIn("strata", json.dumps(meta).lower())
        self.assertEqual(hashlib.sha256(m.read("hc8")).digest(), m.sections["hc8"][2])

    def test_meta(self):
        convert.convert(self.args())
        meta = json.loads(ModelFile(str(self.t / "out.basalt")).read("meta"))
        self.assertEqual(meta["format"], "basalt")
        self.assertEqual(meta["name"], "Tiny")
        self.assertEqual(meta["gguf_kv"]["qwen.block_count"], 2)
        self.assertEqual(meta["gguf_kv"]["qwen.rope.freqs"], [0.5, 1.5])
        self.assertEqual(meta["gguf_kv"]["qwen.nan"], "inf")
        self.assertEqual(meta["tensors"]["token_embd.weight"],
                         {"type": "F32", "type_id": 0, "shape": [3, 8], "bytes": 96, "offset": 0})
        self.assertEqual(meta["tensors"]["output_norm.weight"]["offset"], 4096)
        self.assertEqual(set(meta["tensors"]), {"token_embd.weight", "output_norm.weight"})
        self.assertEqual(meta["profile"]["n_pairs"], 6)
        self.assertEqual(meta["hc8"], {"bytes": 123})
        self.assertEqual(meta["vision"]["mmproj"], "vision/mmproj")
        self.assertEqual(meta["sources"][0]["size"], self.gguf.stat().st_size)
        self.assertIn("engine_args", meta)
        self.assertEqual(meta["sampling"]["top_k"], 20)

    def test_no_vision_and_overwrite(self):
        convert.convert(self.args(no_vision=True))
        self.assertNotIn("vision/mmproj", ModelFile(str(self.t / "out.basalt")).names())
        with self.assertRaises(SystemExit):
            convert.convert(self.args())
        convert.convert(self.args(force=True))
        self.assertIn("vision/mmproj", ModelFile(str(self.t / "out.basalt")).names())

    def test_vision_sections(self):
        src = self.t / "extra.bin"
        src.write_bytes(b"0123456789")
        self.vision = lambda mm, nat: [("vision/a", b"inline", None, 0, 6), ("vision/b", None, str(src), 2, 5)]
        convert.convert(self.args())
        m = ModelFile(str(self.t / "out.basalt"))
        self.assertEqual(m.read("vision/a"), b"inline")
        self.assertEqual(m.read("vision/b"), b"23456")
        self.assertTrue(m.verify("vision/b"))
        self.assertEqual(json.loads(m.read("meta"))["vision"]["extra"], ["vision/a", "vision/b"])

    def test_bad_profile(self):
        (self.t / "profile.bin").write_bytes(self.profile[:-4])
        with self.assertRaises(SystemExit):
            convert.convert(self.args())
        self.assertFalse((self.t / "out.basalt").exists())


if __name__ == "__main__":
    unittest.main()
