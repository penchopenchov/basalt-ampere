"""tools/_paths.py - where the tools find llama.cpp's gguf-py (for the MTP packer and the fixtures).

It lives in a llama.cpp checkout at <repo>/third_party/llama.cpp.
"""
from __future__ import annotations

import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]


def gguf_py() -> str:
    p = ENGINE / "third_party" / "llama.cpp" / "gguf-py"
    if (p / "gguf").is_dir():
        return str(p)
    sys.exit("llama.cpp's gguf-py was not found: put a llama.cpp checkout in third_party/llama.cpp")


def add_gguf_py() -> None:
    p = gguf_py()
    if p not in sys.path:
        sys.path.insert(0, p)
