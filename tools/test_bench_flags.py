"""Every flag the engine lists in its help reaches the engine through basalt-bench.

    BASALT_BIN=bin/basalt BASALT_BENCH_BIN=bin/basalt-bench python3 -m unittest tools.test_bench_flags
"""
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = Path(os.environ.get("BASALT_BIN", ROOT / "bin" / "basalt"))
BENCH = Path(os.environ.get("BASALT_BENCH_BIN", ROOT / "bin" / "basalt-bench"))
# bench's own options that share an engine flag's name
BENCH_OWNED = {"--help", "--model", "--engine", "--spec", "--spec-min-p", "--set"}


def engine_flags() -> list[str]:
    out = subprocess.run([str(ENGINE), "--help"], capture_output=True, text=True).stderr
    return sorted(set(re.findall(r"^\s+(--[a-z0-9][a-z0-9-]*)", out, re.M)))


@unittest.skipUnless(ENGINE.is_file() and BENCH.is_file(), "build bin/basalt and bin/basalt-bench first")
class BenchFlags(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        for n in ("conv", "struct", "prose"):
            (d / f"{n}.ids").write_text("1,2,3\n")
        (d / "prefill.ids").write_text("1,2\n3,4,5\n6\n")
        self.fixtures = d

    def tearDown(self):
        self.tmp.cleanup()

    def command(self, *extra: str) -> list[str]:
        r = subprocess.run([str(BENCH), "--engine", str(ENGINE), "--model", "m.basalt", "--fixtures", str(self.fixtures),
                            "--print-command", *extra], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.split()

    def test_the_flag_list_is_found(self):
        self.assertGreater(len(engine_flags()), 100)

    def test_every_engine_flag_reaches_the_engine(self):
        for flag in engine_flags():
            if flag in BENCH_OWNED:
                continue
            with self.subTest(flag=flag):
                cmd = self.command(flag, "1")
                self.assertIn(flag, cmd)

    def test_a_flag_replaces_the_built_in(self):
        cmd = self.command("--max-context", "999999", "--kv", "int8")
        self.assertEqual(cmd[cmd.index("--max-context") + 1], "999999")
        self.assertEqual(cmd[cmd.index("--kv") + 1], "int8")
        self.assertEqual(cmd.count("--max-context"), 1)

    def test_built_ins_are_minimal(self):
        cmd = self.command()
        self.assertEqual(cmd[cmd.index("--spec") + 1], "0")
        self.assertNotIn("--kv", cmd)
        self.assertNotIn("--resident-experts", cmd)

    def test_help_lists_the_engine_flags(self):
        r = subprocess.run([str(BENCH), "--help", "--engine", str(ENGINE)], capture_output=True, text=True)
        self.assertIn("Engine flags (passed to basalt):", r.stdout)
        self.assertIn("--max-context", r.stdout)


if __name__ == "__main__":
    unittest.main()
