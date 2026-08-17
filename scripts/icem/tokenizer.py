"""Deterministic tokenization with character offsets for release spans."""

from __future__ import annotations

import re

from .schema import Token


TOKEN_PATTERN = re.compile(
    r"""
    https?://[^\s]+
    | www\.[^\s]+
    | [A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}
    | \[[A-Z][A-Z0-9_:-]*\]
    | @[A-Za-z0-9_]{2,32}
    | \#\w+
    | [A-Za-z]+(?:['-][A-Za-z]+)*
    | \d+(?:[.,:/-]\d+)*
    | \S
    """,
    re.VERBOSE,
)


def tokenize_with_offsets(text: str) -> tuple[Token, ...]:
    """Return deterministic surface tokens with source character offsets."""

    return tuple(
        Token(index=index, text=match.group(0), start=match.start(), end=match.end())
        for index, match in enumerate(TOKEN_PATTERN.finditer(text))
    )
