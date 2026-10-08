"""What the engine log says while it starts: load progress bars, warnings and errors, and start-failure hints."""
from __future__ import annotations

import sys
import threading
import time


DRAFT_HEAD_FAIL = "the draft head does not fit"
DRAFT_HEAD_HINT = "free VRAM, or run without the drafter (--spec 0)"


def start_failure_hint(log: str | None, offset: int) -> str:
    """What to change when the engine stopped at the start because the MTP draft head did not fit the VRAM
    left: the engine's own `basalt mtp:` lines after that failure (what it needs, what is free, the smaller
    subsets), else the same advice in words for an older engine.  "" for any other failure: the log says why."""
    if not log:
        return ""
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    if DRAFT_HEAD_FAIL not in text:
        return ""
    said = [x.strip()[len("basalt mtp: "):] for x in text.splitlines()
            if x.strip().startswith("basalt mtp: ") and ("draft head over" in x or "hint:" in x)]
    return ". mtp: " + DRAFT_HEAD_FAIL + ". " + (" ".join(said) if said else "Hint: " + DRAFT_HEAD_HINT)


def start_last_line(log: str | None, offset: int) -> str:
    """The last meaningful line this start wrote to the engine log (without its `basalt generate:` prefix), for the
    error when the engine ended before READY without an error line of its own.  "" when there is none."""
    if not log:
        return ""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            start = max(offset, f.tell() - 64 * 1024)   # never an earlier start's lines
            f.seek(start)
            text = f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    lines = text.splitlines()[1 if start > offset else 0:]   # not a line cut in half
    for x in reversed(lines):
        x = x.strip()
        if x and not x.startswith("basalt warning:"):
            return x.removeprefix("basalt generate: ").removeprefix("basalt serve: ")
    return ""


class LoadBars:
    """The engine's `basalt load:` lines, shown as one bar per target, each against that target's own planned size
    (`plan`).  A target appears when it starts loading, below the ones that started before it; one without a plan
    yet shows a byte counter.  A `copy` target (experts another tier already holds, written to a second card) is counted apart
    and never goes into the total.  At the end (finish) the bars become one line with the size, time and split
    across targets; off a terminal only that line is printed."""

    WIDTH = 24

    def __init__(self, out=None, gpus=None):
        self.out = out or sys.stdout
        self.gpus = list(gpus or [])                   # CUDA ordinal -> nvidia-smi number, when --gpu picked cards
        self.tty = self.out.isatty()
        self.total = 0                                 # the announced bytes of the whole start (0: not announced)
        self.targets: dict[tuple[str, bool], dict] = {}   # (target, copy) -> {plan, done, last, seen, start}
        self.started = 0                               # targets that have begun loading
        self.t0 = None
        self.drawn = 0                                 # lines of bars on screen
        self.kv: list[tuple[str, int, int]] = []       # (target, bytes, tokens): allocated, not loaded

    def name(self, target: str) -> str:
        if target == "ram":
            return "system ram"
        if target.startswith("cuda") and target[4:].isdigit() and int(target[4:]) < len(self.gpus):
            return f"cuda{self.gpus[int(target[4:])]}"
        return target

    def _target(self, target: str, copy: bool) -> dict:
        return self.targets.setdefault((target, copy), {"plan": 0, "done": 0, "last": 0, "seen": 0, "start": None})

    def done(self) -> int:
        return sum(t["done"] + t["last"] for (_, c), t in self.targets.items() if not c)

    @staticmethod
    def _bar(done: int, total: int, width: int) -> str:
        f = min(1.0, done / total) if total > 0 else 0.0
        n = int(f * width)
        return "=" * n + (">" if n < width else "") + " " * (width - n - 1)

    def lines(self) -> list[str]:
        rows = []
        started = [kv for kv in self.targets.items() if kv[1]["start"] is not None]
        for (target, copy), t in sorted(started, key=lambda kv: kv[1]["start"]):
            have = t["done"] + t["last"]
            if t["plan"] <= 0 and not have:
                continue                               # planned to hold nothing
            verb = "copying experts to" if copy else "loading weights into"
            rows.append((f"[server] {verb} {self.name(target)}", have, t["plan"]))
        pad = max((len(r[0]) for r in rows), default=0)
        out = []
        for label, have, plan in rows:
            if plan <= 0:                              # no plan yet: a count, not a bar
                out.append(f"{label.ljust(pad)}  {have / 2**30:5.1f} GiB")
            else:
                plan = max(plan, have)
                out.append(f"{label.ljust(pad)} [{self._bar(have, plan, self.WIDTH)}] "
                           f"{have / 2**30:5.1f}/{plan / 2**30:.1f} GiB")
        return out

    def _erase(self):
        if self.drawn:
            self.out.write("\x1b[1A\x1b[2K" * self.drawn)
            self.drawn = 0

    def _draw(self):
        if self.tty:
            rows = self.lines()
            for r in rows:
                self.out.write("\x1b[2K" + r + "\n")
            self.drawn = len(rows)
        self.out.flush()

    def say(self, text: str) -> None:
        """A line printed above the bars."""
        self._erase()
        self.out.write(text + "\n")
        self._draw()

    def set_kv(self, target: str, nbytes: int, tokens: int) -> None:
        self.kv.append((target, nbytes, tokens))

    def set_total(self, total: int) -> None:
        """The bytes the whole start places (the engine may lower it when part is left on the file)."""
        self.total = total

    def set_plan(self, target: str, plan: int, copy: bool = False) -> None:
        if self.t0 is None:
            self.t0 = time.time()
        self._target(target, copy)["plan"] = plan
        self._erase()
        self._draw()

    def update(self, target: str, done: int, total: int, copy: bool = False) -> None:
        if self.t0 is None:
            self.t0 = time.time()
        t = self._target(target, copy)
        if t["start"] is None:
            t["start"] = self.started
            self.started += 1
        done = min(done, total)
        if t["seen"] != total or done < t["last"]:     # a new phase of this target: the last one is complete
            t["done"] += t["last"] if t["seen"] else 0
            t["last"] = 0
            t["seen"] = total
        t["last"] = done
        self._erase()
        self._draw()

    def finish(self) -> None:
        """The engine is up: the bars become the result line."""
        self._erase()
        done = self.done()
        want = self.total or sum(t["plan"] for (_, c), t in self.targets.items() if not c)
        if done > 0 and done >= want:                  # a start that failed mid-load says so elsewhere
            def part(copy):
                return {tg: t["done"] + t["last"] for (tg, c), t in self.targets.items()
                        if c == copy and t["done"] + t["last"]}
            parts = ", ".join(f"{self.name(tg)} {b / 2**30:.1f} GiB" for tg, b in part(False).items())
            extra = "".join(f"; {self.name(tg)} copy {b / 2**30:.1f} GiB" for tg, b in part(True).items())
            self.out.write(f"[server] loaded {done / 2**30:.1f} GiB in {time.time() - self.t0:.1f} s "
                           f"({parts}{extra})\n")
        for target, nbytes, tokens in self.kv:
            self.out.write(f"[server] kv cache: {nbytes / 2**30:.1f} GiB on {self.name(target)}, {tokens} tokens\n")
        self.kv.clear()
        self.targets.clear()
        self.out.flush()


def load_progress(line: str):
    """(target, done, total, copy) from an engine `basalt load: [copy ]<target> <done> <total>` line, else None."""
    if not line.startswith("basalt load: "):
        return None
    parts = line.split()[2:]
    copy = parts[:1] == ["copy"]
    if copy:
        parts = parts[1:]
    if len(parts) != 3 or parts[0] in ("total", "plan", "kv"):
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2]), copy
    except ValueError:
        return None


def load_plan(line: str):
    """(target, bytes, copy) from an engine `basalt load: plan [copy ]<target> <bytes>` line, else None."""
    parts = line.split()
    if parts[:3] != ["basalt", "load:", "plan"]:
        return None
    parts = parts[3:]
    copy = parts[:1] == ["copy"]
    if copy:
        parts = parts[1:]
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1]), copy


def load_kv(line: str):
    """(target, bytes, tokens) from an engine `basalt load: kv <target> <bytes> <tokens>` line, else None."""
    parts = line.split()
    if len(parts) == 6 and parts[:3] == ["basalt", "load:", "kv"] and parts[4].isdigit() and parts[5].isdigit():
        return parts[3], int(parts[4]), int(parts[5])
    return None


def load_total(line: str):
    """The bytes of the whole start from an engine `basalt load: total <bytes>` line, else None."""
    parts = line.split()
    if len(parts) == 4 and parts[:3] == ["basalt", "load:", "total"] and parts[3].isdigit():
        return int(parts[3])
    return None


def engine_warning(line: str):
    """The text of an engine `basalt warning: <text>` line, else None."""
    if line.startswith("basalt warning: ") and line[16:].strip():
        return line[16:].strip()
    return None


def engine_error(line: str):
    """The text of an engine `basalt error: <text>` line, else None."""
    if line.startswith("basalt error: ") and line[14:].strip():
        return line[14:].strip()
    return None


def start_errors(log: str | None, offset: int) -> list[str]:
    """The engine's `basalt error:` lines this start wrote to its log."""
    if not log:
        return []
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            lines = f.read().decode("utf-8", "replace").splitlines()
    except (OSError, ValueError):
        return []
    return [e for x in lines if (e := engine_error(x.strip())) is not None]


def narrate_start(log_path: str, offset: int, done: threading.Event, heartbeat=20.0, out=None, gpus=None) -> None:
    """Show the engine's load progress from its log (LoadBars) and its warnings, and a heartbeat while nothing
    moves."""
    bars = LoadBars(out, gpus)
    t0 = last = time.time()
    pos = offset
    while True:
        stop = done.wait(0.25)
        try:
            with open(log_path, "rb") as f:
                f.seek(pos)
                chunk = f.read()
        except OSError:
            chunk = b""
        if chunk.count(b"\n"):
            cut = chunk.rfind(b"\n") + 1
            pos += cut
            for line in chunk[:cut].decode("utf-8", "replace").splitlines():
                if (n := load_total(line)) is not None:
                    bars.set_total(n)
                    last = time.time()
                elif (kv := load_kv(line)) is not None:
                    bars.set_kv(*kv)
                elif (pl := load_plan(line)) is not None:
                    bars.set_plan(*pl)
                    last = time.time()
                elif (p := load_progress(line)) is not None:
                    bars.update(*p)
                    last = time.time()
                elif (w := engine_warning(line)) is not None:
                    bars.say(f"[server] warning: {w}")
        if stop:
            bars.finish()
            return
        if time.time() - last > heartbeat:
            last = time.time()
            bars.say(f"[server] still starting, {time.time() - t0:.0f} s")
