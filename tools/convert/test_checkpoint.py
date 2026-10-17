"""Tests for checkpoint.py on small synthetic safetensors files.

    python3 -m unittest discover -s tools/convert -p 'test_checkpoint.py'"""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import checkpoint as C  # noqa: E402
import model_sections as S  # noqa: E402

E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def bf16(x: np.ndarray) -> np.ndarray:
    """float32 values that are exactly BF16 -> their u16 codes."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    assert not np.count_nonzero(u & 0xFFFF)
    return (u >> 16).astype("<u2")


def write_safetensors(path: Path, tensors: dict) -> None:
    """name -> (dtype, shape, raw bytes), in the given order."""
    head, blobs, at = {}, [], 0
    for k, (dt, shape, raw) in tensors.items():
        head[k] = {"dtype": dt, "shape": list(shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    hb = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def ue4m3(x) -> float:
    x = int(x)
    e, m = (x >> 3) & 15, x & 7
    return m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)


def modelopt_values(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """What Model Optimizer's NVFP4 rows mean: value 2i is the low nibble of byte i, 2i+1 the high one."""
    rows = codes.shape[0]
    nib = np.empty((rows, codes.shape[1] * 2), np.uint8)
    nib[:, 0::2], nib[:, 1::2] = codes & 15, codes >> 4
    mag = np.array(E2M1, np.float64)[nib & 7] * np.where(nib & 8, -1, 1)
    return mag * np.vectorize(ue4m3)(scales).repeat(16, axis=1)


def basalt_values(blocks: np.ndarray, cols: int) -> np.ndarray:
    """Basalt's NVFP4 block decode (dequantize_nvfp4): sub-block s reads byte 8 s + j, low nibble value j,
    high nibble value j + 8."""
    b = blocks.reshape(-1, cols // 64, 36)
    out = np.empty((b.shape[0], cols // 64, 64))
    for s in range(4):
        sc = np.vectorize(ue4m3)(b[:, :, s])
        q = b[:, :, 4 + 8 * s:12 + 8 * s]
        for half, nib in ((0, q & 15), (8, q >> 4)):
            out[:, :, 16 * s + half:16 * s + half + 8] = (np.array(E2M1)[nib & 7] * np.where(nib & 8, -1, 1)) * sc[..., None]
    return out.reshape(b.shape[0], cols)


class Nvfp4RepackTest(unittest.TestCase):
    def test_values_survive_the_repack(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 256, size=(5, 64), dtype=np.uint8)            # 5 rows of 128 values
        scales = rng.integers(1, 0x7F, size=(5, 8), dtype=np.uint8)
        got = C.nvfp4_rows(codes, scales)
        self.assertEqual(got.size, 5 * S.nvfp4_row_bytes(128))
        np.testing.assert_array_equal(basalt_values(got, 128), modelopt_values(codes, scales))

    def test_expert_blob_and_tail(self):
        rng = np.random.default_rng(1)
        h, ff = 128, 64
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.safetensors"
            tens, want = {}, []
            shapes = {"gate_proj": (ff, h), "up_proj": (ff, h), "down_proj": (h, ff)}
            for i, (r, (rows, cols)) in enumerate(shapes.items()):
                w = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
                s = rng.integers(1, 0x7F, size=(rows, cols // 16), dtype=np.uint8)
                tens[f"{r}.weight"] = ("U8", w.shape, w.tobytes())
                tens[f"{r}.weight_scale"] = ("F8_E4M3", s.shape, s.tobytes())
                tens[f"{r}.weight_scale_2"] = ("F32", (), np.float32(0.5 + i).tobytes())
                want.append(C.nvfp4_rows(w, s).tobytes())
            write_safetensors(p, tens)
            head, base = S.read_safetensors_header(p)
            ent = []
            for r, (rows, cols) in shapes.items():
                ent.append(((p, base + head[f"{r}.weight"]["data_offsets"][0], [rows, cols // 2]),
                            (p, base + head[f"{r}.weight_scale"]["data_offsets"][0], [rows, cols // 16]),
                            (p, base + head[f"{r}.weight_scale_2"]["data_offsets"][0])))
            blob = C._expert_job(([ent, ent], 2))
            one = b"".join(want) + np.array([0.5, 1.5, 2.5, 0], "<f4").tobytes()
            self.assertEqual(blob, one + one)
            # a scale byte with the sign bit set is refused
            raw = bytearray(p.read_bytes())
            raw[base + head["up_proj.weight_scale"]["data_offsets"][0]] |= 0x80
            p.write_bytes(bytes(raw))
            self.assertIsInstance(C._expert_job(([ent], 1)), str)


GDN = 2 * S.GDN_K_HEADS * S.GDN_HEAD + S.GDN_V_HEADS * S.GDN_HEAD      # q, k, then the value channels


class CheckpointDenseTest(unittest.TestCase):
    """A one-layer checkpoint at the real GDN head geometry with narrow matrices."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        rng = np.random.default_rng(2)
        H = 256
        self.v = {}

        def t(name, shape, scale=1.0):
            x = (rng.integers(-64, 64, size=shape) * scale / 64).astype(np.float32)
            self.v[name] = x
            return ("BF16", shape, bf16(x).tobytes())

        L = "model.language_model.layers.0."
        tens = {
            L + "linear_attn.A_log": t(L + "linear_attn.A_log", (48,)),
            L + "linear_attn.conv1d.weight": t(L + "linear_attn.conv1d.weight", (GDN, 1, 4)),
            L + "linear_attn.in_proj_a.weight": t(L + "linear_attn.in_proj_a.weight", (48, H)),
            L + "linear_attn.in_proj_z.weight": t(L + "linear_attn.in_proj_z.weight", (48 * 128, H)),
            L + "linear_attn.norm.weight": t(L + "linear_attn.norm.weight", (128,)),
            L + "linear_attn.out_proj.weight": t(L + "linear_attn.out_proj.weight", (H, 48 * 128)),
            L + "attn_hyper_connection.hc_norm.weight": t(L + "attn_hyper_connection.hc_norm.weight", (H,)),
            L + "mlp.shared_expert.down_proj.weight": t(L + "mlp.shared_expert.down_proj.weight", (H, 640), 0.0625),
            L + "self_attn.indexer.index_qk_proj.weight": t(L + "self_attn.indexer.index_qk_proj.weight", (640, H)),
            L + "mlp.experts.0.gate_proj.weight": ("U8", (2, 2), b"\0" * 4),
            "mtp.layers.0.mlp.gate.weight": ("BF16", (1,), b"\0\0"),
        }
        write_safetensors(d / "model-00001-of-00001.safetensors", tens)
        cfg = {"text_config": {"linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
                               "linear_value_head_dim": 128, "indexer_n_heads": 4, "indexer_head_dim": 128}}
        (d / "config.json").write_text(json.dumps(cfg))
        self.d, self.L = d, L

    def tearDown(self):
        self.tmp.cleanup()

    def test_names_order_types(self):
        dn = C.CheckpointDense(C.Checkpoint(self.d), q6k=True)
        self.assertEqual(list(dn.tensors), [
            "blk.0.ssm_a", "blk.0.ssm_conv1d.weight", "blk.0.ssm_alpha.weight", "blk.0.attn_gate.weight",
            "blk.0.ssm_norm.weight", "blk.0.ssm_out.weight", "blk.0.hc_attn_norm.weight",
            "blk.0.ffn_down_shexp.weight", "blk.0.indexer.q_proj.weight", "blk.0.indexer.k_proj.weight"])
        ty = {n: (t.type_name, t.ne) for n, t in dn.tensors.items()}
        self.assertEqual(ty["blk.0.attn_gate.weight"], ("Q6_K", [256, 6144]))
        self.assertEqual(ty["blk.0.ffn_down_shexp.weight"], ("Q8_0", [640, 256]))    # 640-value rows: not Q6_K
        self.assertEqual(ty["blk.0.ssm_conv1d.weight"], ("F32", [4, GDN]))
        self.assertEqual(ty["blk.0.indexer.k_proj.weight"], ("BF16", [256, 128]))
        q8 = C.CheckpointDense(C.Checkpoint(self.d), q6k=False)
        self.assertEqual(q8.tensors["blk.0.attn_gate.weight"].type_name, "Q8_0")

    def test_transforms(self):
        dn = C.CheckpointDense(C.Checkpoint(self.d), q6k=True)
        L, v = self.L, self.v
        heads = S.gdn_block_source()
        f32 = lambda n: dn.bytes(n).view("<f4")
        a = v[L + "linear_attn.A_log"]
        np.testing.assert_array_equal(f32("blk.0.ssm_a"), (-np.exp(a.astype(np.float64))).astype(np.float32)[heads])
        np.testing.assert_array_equal(f32("blk.0.hc_attn_norm.weight"), v[L + "attn_hyper_connection.hc_norm.weight"] + 1)
        np.testing.assert_array_equal(f32("blk.0.ssm_norm.weight"), v[L + "linear_attn.norm.weight"])
        conv = v[L + "linear_attn.conv1d.weight"][:, 0, :]
        np.testing.assert_array_equal(f32("blk.0.ssm_conv1d.weight").reshape(GDN, 4), conv[S.gdn_index(GDN, 4096)])
        self.assertEqual(dn.bytes("blk.0.ssm_alpha.weight").tobytes(),
                         bf16(v[L + "linear_attn.in_proj_a.weight"][heads]).tobytes())
        qk = v[L + "self_attn.indexer.index_qk_proj.weight"]
        self.assertEqual(dn.bytes("blk.0.indexer.k_proj.weight").tobytes(), bf16(qk[512:]).tobytes())

    def test_quantized_streams(self):
        import multiprocessing
        dn = C.CheckpointDense(C.Checkpoint(self.d), q6k=True)
        L, v = self.L, self.v
        with multiprocessing.Pool(2) as pool:
            z = b"".join(dn.chunks("blk.0.attn_gate.weight", pool))
            o = b"".join(dn.chunks("blk.0.ssm_out.weight", pool))
            dw = b"".join(dn.chunks("blk.0.ffn_down_shexp.weight", pool))
        zv = v[L + "linear_attn.in_proj_z.weight"][S.gdn_index(6144, 0)]
        self.assertEqual(z, S.q6_k(zv.reshape(-1)))
        ov = v[L + "linear_attn.out_proj.weight"][:, S.gdn_index(6144, 0)]
        self.assertEqual(o, S.q6_k(np.ascontiguousarray(ov).reshape(-1)))
        self.assertEqual(dw, S.q8_0(v[L + "mlp.shared_expert.down_proj.weight"].reshape(-1)))
        self.assertEqual(len(z), dn.tensors["blk.0.attn_gate.weight"].expected_bytes())

    def test_unknown_tensor_is_refused(self):
        p = self.d / "model-00001-of-00001.safetensors"
        head, base = S.read_safetensors_header(p)
        tens = {k: (h["dtype"], h["shape"], p.read_bytes()[base + h["data_offsets"][0]:base + h["data_offsets"][1]])
                for k, h in head.items()}
        tens[self.L + "linear_attn.new_thing.weight"] = ("BF16", (2,), b"\0" * 4)
        write_safetensors(p, tens)
        with self.assertRaises(S.ConvertError):
            C.CheckpointDense(C.Checkpoint(self.d))


class TruncationTest(unittest.TestCase):
    def test_short_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.safetensors"
            write_safetensors(p, {"a": ("BF16", (4,), b"\1" * 8), "b": ("F32", (2,), b"\2" * 8)})
            self.assertEqual(S.read_safetensors_header(p)[0]["b"]["data_offsets"], [8, 16])
            full = p.read_bytes()
            for cut in (1, 9):
                p.write_bytes(full[:-cut])
                with self.assertRaisesRegex(S.ConvertError, "truncated"):
                    S.read_safetensors_header(p)
            p.write_bytes(full[:20])                       # inside the header
            with self.assertRaisesRegex(S.ConvertError, "truncated"):
                S.read_safetensors_header(p)
            p.write_bytes(full[:5])
            with self.assertRaises(S.ConvertError):
                S.read_safetensors_header(p)


class TokenizerKvTest(unittest.TestCase):
    def test_types_gaps_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            vocab = {"a": 0, "b": 1, "ab": 2}
            tj = {"model": {"type": "BPE", "vocab": vocab, "merges": [["a", "b"]]},
                  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                      {"type": "Split", "pattern": {"Regex": C.QWEN35_HF_SPLIT}}, {"type": "ByteLevel"}]},
                  "added_tokens": [{"id": 4, "content": "<|end|>", "special": True},
                                   {"id": 5, "content": "<|fim|>", "special": False},
                                   {"id": 6, "content": "<think>", "special": False}]}
            (d / "tokenizer.json").write_text(json.dumps(tj))
            (d / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|end|>", "pad_token": "<|end|>",
                                                                 "bos_token": None, "add_bos_token": False}))
            (d / "generation_config.json").write_text(json.dumps({"bos_token_id": 4}))
            (d / "chat_template.jinja").write_text("{{ x }}")
            (d / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 8}}))
            (d / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", 2) + b"{}")
            kv = C.tokenizer_kv(C.Checkpoint(d))
            self.assertEqual(kv["tokenizer.ggml.tokens"], ["a", "b", "ab", "[PAD3]", "<|end|>", "<|fim|>", "<think>", "[PAD7]"])
            self.assertEqual(kv["tokenizer.ggml.token_type"], [1, 1, 1, 5, 3, 3, 4, 5])
            self.assertEqual(kv["tokenizer.ggml.merges"], ["a b"])
            self.assertEqual([kv[f"tokenizer.ggml.{k}_token_id"] for k in ("eos", "padding", "bos")], [4, 4, 4])
            self.assertEqual(kv["tokenizer.chat_template"], "{{ x }}")
            tj["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = r"\s+"
            (d / "tokenizer.json").write_text(json.dumps(tj))
            with self.assertRaises(S.ConvertError):
                C.tokenizer_kv(C.Checkpoint(d))


if __name__ == "__main__":
    unittest.main()
