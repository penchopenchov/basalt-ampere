"""--model FILE.basalt: the config derived from the file, the tokenizer unpacked once, the peer card, vision."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from serve import modelfile
from serve.server import engine_args, gpus_in_play
from model_file import ModelFile, ModelWriter  # noqa: E402

META = {"name": "tiny-basalt", "engine_args": ["--serve", "--context", "4096"],
        "sampling": {"temperature": 0.7, "top_p": 0.9}}
TOKENIZER = {"vocab.json": b'{"a": 0}', "merges.txt": b"a a", "token_type.json": b"[1]",
             "chat_template.jinja": b"{{ messages }}"}


class Tmp(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.dir = Path(self.td.name)
        patch = mock.patch.dict(os.environ, {"BASALT_CACHE": str(self.dir / "cache")})
        patch.start()
        self.addCleanup(patch.stop)
        self.exe = self.dir / "bin" / "basalt"
        self.exe.parent.mkdir()
        self.exe.write_text("")

    def build(self, vision=False, meta=META):
        path = self.dir / "tiny.basalt"
        w = ModelWriter(str(path))
        w.add_bytes("meta", json.dumps(meta).encode())
        for n, b in TOKENIZER.items():
            w.add_bytes("tokenizer/" + n, b)
        if vision:
            w.add_bytes("vision/mmproj", b"MMPROJ" * 100)
            w.add_bytes("vision/text_vocab", b"VOCAB")
        w.close()
        return str(path)


class Derivation(Tmp):
    def test_config(self):
        path = self.build()
        cfg = modelfile.model_config(path, str(self.exe), gpus=1)
        self.assertEqual(cfg["exe"], str(self.exe))
        self.assertEqual(cfg["args"], ["--model", path, "--serve", "--context", "4096"])
        self.assertEqual(cfg["model_name"], "tiny-basalt")
        self.assertEqual(cfg["sampling"], META["sampling"])
        self.assertTrue(cfg["log"].startswith(str(self.dir / "cache")))
        self.assertNotIn("vision", cfg)
        self.assertEqual(engine_args(cfg), cfg["args"])

    def test_cache_key_is_the_meta_hash(self):
        path = self.build()
        key = ModelFile(path).sections["meta"][2].hex()[:16]
        tok = Path(modelfile.model_config(path, str(self.exe), gpus=1)["tokenizer"])
        self.assertEqual(tok, self.dir / "cache" / key / "tokenizer")

    def test_tokenizer_extracted_once_and_reused(self):
        path = self.build()
        tok = Path(modelfile.model_config(path, str(self.exe), gpus=1)["tokenizer"])
        for n, b in TOKENIZER.items():
            self.assertEqual((tok / n).read_bytes(), b)
        self.assertEqual([p.name for p in tok.parent.iterdir()], ["tokenizer"])
        (tok / "vocab.json").write_bytes(b"{}")                # a present directory is not rewritten
        modelfile.model_config(path, str(self.exe), gpus=1)
        self.assertEqual((tok / "vocab.json").read_bytes(), b"{}")

    def test_missing_tokenizer_and_bad_file(self):
        w = ModelWriter(str(self.dir / "bare.basalt"))
        w.add_bytes("meta", b"{}")
        w.close()
        with self.assertRaises(ValueError):
            modelfile.model_config(str(self.dir / "bare.basalt"), str(self.exe), gpus=1)
        (self.dir / "junk").write_bytes(b"x" * 5000)
        with self.assertRaises(ValueError):
            modelfile.model_config(str(self.dir / "junk"), str(self.exe), gpus=1)


class Peer(Tmp):
    def test_second_gpu_becomes_the_peer(self):
        path = self.build()
        cfg = modelfile.model_config(path, str(self.exe), gpus=2)
        self.assertEqual(cfg["args"][-2:], ["--peer-device", "1"])
        self.assertEqual(gpus_in_play(cfg, cfg["args"]), [0, 1])
        self.assertNotIn("--peer-device", modelfile.model_config(path, str(self.exe), gpus=1)["args"])

    def test_vision_tells_the_engine(self):
        path = self.build(vision=True)
        (self.exe.parent / "basalt-vision").write_text("")
        cfg = modelfile.model_config(path, str(self.exe), gpus=1)
        self.assertIn("vision", cfg)
        self.assertIn("--vision", cfg["args"])

    def test_overrides_replace_or_add(self):
        path = self.build(meta={**META, "engine_args": ["--spec", "7", "--spec-min-p", "0.5"]})
        args = modelfile.model_config(path, str(self.exe), gpus=1,
                                      overrides={"--spec": 0, "--mtp-max-t": 4})["args"]
        self.assertEqual(args, ["--model", path, "--spec", "0", "--spec-min-p", "0.5", "--mtp-max-t", "4"])

    def test_file_args_win(self):
        path = self.build(meta={**META, "engine_args": ["--peer-device", "-1"]})
        args = modelfile.model_config(path, str(self.exe), gpus=2)["args"]
        self.assertEqual(args.count("--peer-device"), 1)

    def test_count_respects_visible_devices_and_nvidia_smi(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            self.assertEqual(modelfile.gpu_count(), 1)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            self.assertEqual(modelfile.gpu_count(), 2)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
            self.assertEqual(modelfile.gpu_count(), 0)
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        out = mock.Mock(stdout="GPU 0: RTX (UUID: a)\nGPU 1: RTX (UUID: b)\n")
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("subprocess.run", return_value=out):
            self.assertEqual(modelfile.gpu_count(), 2)
            path = self.build()
            self.assertEqual(modelfile.model_config(path, str(self.exe))["args"][-2:], ["--peer-device", "1"])
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("subprocess.run", side_effect=OSError):
            self.assertEqual(modelfile.gpu_count(), 0)


class Vision(Tmp):
    def test_sidecar_beside_the_exe_and_sections_unpacked(self):
        path = self.build(vision=True)
        side = self.exe.parent / "basalt-vision"
        side.write_text("")
        v = modelfile.model_config(path, str(self.exe), gpus=1)["vision"]
        self.assertEqual(v["exe"], str(side.resolve()))
        self.assertEqual(Path(v["mmproj"]).read_bytes(), b"MMPROJ" * 100)
        self.assertEqual(Path(v["model"]).read_bytes(), b"VOCAB")
        self.assertEqual([p.name for p in Path(v["mmproj"]).parent.iterdir() if ".gguf." in p.name], [])

    def test_text_only_when_the_sidecar_is_missing(self):
        path = self.build(vision=True)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cfg = modelfile.model_config(path, str(self.exe), gpus=1)
        self.assertNotIn("vision", cfg)
        self.assertEqual(len(buf.getvalue().strip().splitlines()), 1)
        self.assertIn("text-only", buf.getvalue())


class Cli(Tmp):
    def run_main(self, *argv):
        from serve import server
        with mock.patch.object(sys, "argv", ["server", *argv]), mock.patch.object(server, "Server") as srv:
            srv.side_effect = RuntimeError("stop")
            with self.assertRaises((SystemExit, RuntimeError)) as cm:
                server.main()
        return cm.exception

    def test_model_needs_exe_and_a_valid_gpu(self):
        path = self.build()
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            e = self.run_main("--engine", "engine", "--model", path)
        self.assertEqual(getattr(e, "code", None), 2)
        self.assertIn("--exe", err.getvalue())
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            e = self.run_main("--engine", "engine", "--model", path, "--exe", str(self.exe), "--gpu", "x")
        self.assertEqual(getattr(e, "code", None), 2)


if __name__ == "__main__":
    unittest.main()
