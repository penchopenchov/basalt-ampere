"""Reading engine flags from a list of command-line words."""
from __future__ import annotations


def flag_index(words: list[str], flag: str) -> int | None:
    """The index of the last `flag` in `words` (the engine's parser lets a later one win), or None."""
    return next((i for i in range(len(words) - 1, -1, -1) if words[i] == flag), None)


def flag_value(words: list[str], flag: str) -> str | None:
    """The word after the last `flag`; None when absent or when no word follows."""
    i = flag_index(words, flag)
    return words[i + 1] if i is not None and i + 1 < len(words) else None
