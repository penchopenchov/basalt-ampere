"""Run the server from a .basalt model file: the engine settings the file implies, its tokenizer and the vision
sidecar's inputs, unpacked once into the cache directory."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from model_file import ModelFile  # noqa: E402

VISION_EXE = "basalt-vision"


def cache_root() -> Path:
    return Path(os.environ.get("BASALT_CACHE") or Path.home() / ".cache" / "basalt")


def cache_dir(mf: ModelFile) -> Path:
    """One directory per model: the first 16 hex digits of the meta section's sha256."""
    return cache_root() / mf.sections["meta"][2].hex()[:16]


def write_section(mf: ModelFile, name: str, dest: Path) -> None:
    """Copy a section to `dest` through a temporary name, so a partial file is never taken for a whole one."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    off, size = mf.section(name)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".")
    try:
        with os.fdopen(fd, "wb") as out, open(mf.path, "rb") as src:
            src.seek(off)
            left = size
            while left > 0:
                b = src.read(min(left, 64 << 20))
                if not b:
                    raise IOError(f"{mf.path}: short read in {name}")
                out.write(b)
                left -= len(b)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def extract_tokenizer(mf: ModelFile) -> Path:
    """The `tokenizer/*` sections as a directory (what tools/tokenizer.py and the chat template read); reused
    when already there."""
    out = cache_dir(mf) / "tokenizer"
    if (out / "vocab.json").exists():
        return out
    names = [n for n in mf.names() if n.startswith("tokenizer/")]
    if not any(n == "tokenizer/vocab.json" for n in names):
        raise ValueError(f"{mf.path}: no tokenizer sections")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out.parent, prefix="tokenizer."))
    try:
        for n in names:
            (tmp / n.split("/", 1)[1]).write_bytes(mf.read(n))
        try:
            os.rename(tmp, out)
        except OSError:                          # another process finished first
            if not (out / "vocab.json").exists():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def gpu_count(visible_env: str | None = None) -> int:
    """The NVIDIA GPUs this process can see: CUDA_VISIBLE_DEVICES when set, else what nvidia-smi lists."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_env is None else visible_env
    if v is not None:
        return len([x for x in v.split(",") if x.strip() and x.strip() != "-1"])
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def vision_config(mf: ModelFile, exe: str) -> dict | None:
    """The vision sidecar's config (`basalt-vision` beside the engine binary, the projector and a vocab-only text
    model unpacked from the file), or None with a one-line reason printed when the model runs text-only."""
    side = Path(exe).resolve().parent / VISION_EXE
    if "vision/mmproj" not in mf.sections:
        return None
    if not side.is_file():
        print(f"[server] warning: {side} not found, text-only (build the vision sidecar)", flush=True)
        return None
    if "vision/text_vocab" not in mf.sections:
        print(f"[server] warning: {mf.path} has no vision/text_vocab section, text-only", flush=True)
        return None
    vdir = cache_dir(mf) / "vision"
    mmproj, vocab = vdir / "mmproj.gguf", vdir / "text_vocab.gguf"
    for name, dest in (("vision/mmproj", mmproj), ("vision/text_vocab", vocab)):
        if not dest.exists() or dest.stat().st_size != mf.section(name)[1]:
            write_section(mf, name, dest)
    return {"exe": str(side), "mmproj": str(mmproj), "model": str(vocab)}


def with_arg(args: list[str], flag: str, value) -> list[str]:
    """`args` with `flag value` replaced, or appended when absent."""
    out = list(args)
    if flag in out:
        out[out.index(flag) + 1] = str(value)
    else:
        out += [flag, str(value)]
    return out


def model_config(path: str, exe: str, gpus: int | None = None, vision: bool = True,
                 overrides: dict | None = None) -> dict:
    """The engine settings the model file implies, with `overrides` ({flag: value}) replacing the file's
    engine arguments.  A second visible GPU becomes the engine's peer expert tier unless they already say."""
    mf = ModelFile(path)
    meta = json.loads(mf.read("meta"))
    path = os.path.abspath(path)
    args = ["--model", path] + [str(a) for a in meta.get("engine_args") or []]
    for flag, value in (overrides or {}).items():
        args = with_arg(args, flag, value)
    if "--peer-device" not in args and (gpu_count() if gpus is None else gpus) > 1:
        args += ["--peer-device", "1"]
    cdir = cache_dir(mf)
    cdir.mkdir(parents=True, exist_ok=True)
    cfg = {"exe": os.path.abspath(exe), "args": args, "model_name": meta.get("name") or "basalt",
           "sampling": dict(meta.get("sampling") or {}), "log": str(cdir / "engine.log"),
           "tokenizer": str(extract_tokenizer(mf))}
    v = vision_config(mf, exe) if vision else None
    if v:
        cfg["vision"] = v
    return cfg
