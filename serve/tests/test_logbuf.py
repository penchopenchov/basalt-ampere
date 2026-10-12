"""serve/util/logbuf.py: the line buffer behind GET /api/logs, the stream tee and the file tail."""
import io
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.util.logbuf import CAPACITY, MAX_LINE, ColorStream, LogBuffer, TeeStream, level, tail_file


class Levels(unittest.TestCase):
    def test_words_decide(self):
        for text in ("CUDA error: out of memory", "request Failed", "FATAL: no device", "Traceback (most recent call last):",
                     "the engine died", "abort called", "an Exception was raised"):
            self.assertEqual(level(text), "err", text)
        for text in ("warning: low on RAM", "WARN cache", "request timeout", "retrying in 15 s", "engine stuck",
                     "slow read"):
            self.assertEqual(level(text), "warn", text)
        for text in ("ready", "reading the prompt: 12 of 100 tokens", "errors=0", "slowly", "prefailed"):
            self.assertEqual(level(text), "info", text)
        self.assertEqual(level("a warning and an error"), "err")          # the worse one wins


class Buffer(unittest.TestCase):
    def test_ids_lines_and_fields(self):
        b = LogBuffer()
        b.add("srv", "one\ntwo\n\n   \nthree")
        lines, last = b.since(0)
        self.assertEqual([x["msg"] for x in lines], ["one", "two", "three"])
        self.assertEqual([x["id"] for x in lines], [1, 2, 3])
        self.assertEqual(last, 3)
        first = lines[0]
        self.assertEqual(set(first), {"id", "t", "src", "lvl", "msg"})
        self.assertEqual((first["src"], first["lvl"]), ("srv", "info"))
        self.assertIsInstance(first["t"], float)
        self.assertEqual(first["t"], round(first["t"], 3))
        self.assertEqual(b.since(0)[0][0], first)

    def test_after_and_limit(self):
        b = LogBuffer()
        for i in range(10):
            b.add("eng", f"line {i}")
        lines, last = b.since(7)
        self.assertEqual([x["id"] for x in lines], [8, 9, 10])
        self.assertEqual(b.since(10), ([], 10))
        lines, last = b.since(0, limit=4)                                  # the newest four, oldest first
        self.assertEqual([x["id"] for x in lines], [7, 8, 9, 10])
        self.assertEqual(last, 10)
        self.assertEqual(LogBuffer().since(0), ([], 0))

    def test_ansi_is_removed_and_long_lines_cut(self):
        b = LogBuffer()
        b.add("eng", "\x1b[31mred\x1b[0m and \x1b[1;32mgreen\x1b[0m")
        b.add("eng", "x" * (MAX_LINE + 50))
        lines = b.since(0)[0]
        self.assertEqual(lines[0]["msg"], "red and green")
        self.assertEqual(len(lines[1]["msg"]), MAX_LINE)

    def test_a_ring(self):
        b = LogBuffer()
        for i in range(CAPACITY + 5):
            b.add("srv", f"n{i}")
        lines, last = b.since(0, limit=CAPACITY * 2)
        self.assertEqual((len(lines), last), (CAPACITY, CAPACITY + 5))
        self.assertEqual(lines[0]["id"], 6)

    def test_threads_get_distinct_ids(self):
        b = LogBuffer()
        threads = [threading.Thread(target=lambda: [b.add("srv", "x") for _ in range(100)]) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ids = [x["id"] for x in b.since(0, 1000)[0]]
        self.assertEqual(ids, list(range(1, 801)))


class Tee(unittest.TestCase):
    def test_the_stream_is_unchanged_and_whole_lines_are_kept(self):
        out, b = io.StringIO(), LogBuffer()
        tee = TeeStream(out, b, "srv")
        print("[server] model unloaded", file=tee)
        tee.write("part")
        self.assertEqual([x["msg"] for x in b.since(0)[0]], ["model unloaded"])     # prefix stripped, no partial line
        tee.write("ial\nnext: failed\n")
        self.assertEqual([(x["msg"], x["lvl"], x["src"]) for x in b.since(0)[0]][1:],
                         [("partial", "info", "srv"), ("next: failed", "err", "srv")])
        self.assertEqual(out.getvalue(), "[server] model unloaded\npartial\nnext: failed\n")
        tee.flush()
        self.assertFalse(tee.closed)                                                  # the stream's own attributes


class Tail(unittest.TestCase):
    def wait_for(self, b, count, timeout=5):
        end = time.time() + timeout
        while time.time() < end and len(b.since(0)[0]) < count:
            time.sleep(0.02)

    def test_follows_from_the_end_and_survives_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("an earlier run\n", encoding="utf-8")
            b = LogBuffer()
            threading.Thread(target=tail_file, args=(b, str(log), "eng", 0.02), daemon=True).start()
            time.sleep(0.2)
            with open(log, "a", encoding="utf-8") as f:
                f.write("basalt: ready\nhalf a li")
            self.wait_for(b, 1)
            self.assertEqual([x["msg"] for x in b.since(0)[0]], ["basalt: ready"])
            with open(log, "a", encoding="utf-8") as f:
                f.write("ne\n")
            self.wait_for(b, 2)
            self.assertEqual(b.since(0)[0][1]["msg"], "half a line")
            log.write_text("new run\n", encoding="utf-8")                              # emptied and written again
            self.wait_for(b, 3)
            lines = b.since(0)[0]
            self.assertEqual((lines[2]["msg"], lines[2]["src"]), ("new run", "eng"))

    def test_a_file_that_appears_later(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "later.log"
            b = LogBuffer()
            threading.Thread(target=tail_file, args=(b, str(log), "eng", 0.02), daemon=True).start()
            time.sleep(0.1)
            log.write_text("hello\n", encoding="utf-8")
            self.wait_for(b, 1)
            self.assertEqual(b.since(0)[0][0]["msg"], "hello")


class Tty(io.StringIO):
    def isatty(self):
        return True


LINES = ("[server] warning: model 76.9 GiB\nbasalt error: boom\n[server] refused Host 'x'\n"
         "[server] listening\nbasalt warning: low\n")


class Color(unittest.TestCase):
    def wrapped(self, stream, env=None):
        with mock.patch.dict(os.environ, env or {}, clear=False):
            if env is None:
                os.environ.pop("NO_COLOR", None)
            return ColorStream.wrap(stream)

    def test_plain_off_a_terminal(self):
        out = io.StringIO()
        w = self.wrapped(out)
        self.assertIs(w, out)
        w.write(LINES)
        self.assertEqual(out.getvalue(), LINES)

    def test_terminal_colors_warning_error_refused(self):
        out = Tty()
        w = self.wrapped(out)
        for ch in LINES:                                 # print() style: pieces, newline apart
            w.write(ch)
        w.flush()
        y, r, z = "\x1b[33m", "\x1b[31m", "\x1b[0m"
        self.assertEqual(out.getvalue(), f"{y}[server] warning: model 76.9 GiB{z}\n{r}basalt error: boom{z}\n"
                         f"{r}[server] refused Host 'x'{z}\n[server] listening\n{y}basalt warning: low{z}\n")

    def test_no_color_disables(self):
        out = Tty()
        self.assertIs(self.wrapped(out, {"NO_COLOR": "1"}), out)

    def test_empty_no_color_still_colors(self):
        self.assertIsInstance(self.wrapped(Tty(), {"NO_COLOR": ""}), ColorStream)

    def test_redraw_passes_through_at_once(self):
        out = Tty()
        w = self.wrapped(out)
        w.write("\x1b[1A\x1b[2K")
        w.write("bar 50%\r")
        self.assertEqual(out.getvalue(), "\x1b[1A\x1b[2Kbar 50%\r")

    def test_tee_logs_plain_text(self):
        b = LogBuffer()
        t = TeeStream(self.wrapped(Tty()), b, "srv")
        t.write("[server] warning: x\n")
        self.assertEqual(b.since(0)[0][0]["msg"], "warning: x")
        self.assertTrue(t.isatty())


if __name__ == "__main__":
    unittest.main()
