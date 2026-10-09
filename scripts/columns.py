"""
Deterministic column resolution.

Naive substring matching silently binds the wrong column when several names
share a fragment (e.g. ``Est_mean`` matching a request for ``_mean``). This
resolver walks strictly decreasing levels of confidence and refuses to guess
when a level yields more than one candidate.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence


class ColumnResolutionError(KeyError):
    """Raised when a requested column is missing or ambiguous."""


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_column(
    requested: str,
    available: Iterable[str],
    *,
    allow_fuzzy: bool = True,
    required: bool = True,
) -> Optional[str]:
    """Resolve one requested name against the available columns.

    Resolution order: exact -> case-insensitive exact -> alphanumeric-normalised
    exact -> unique prefix -> unique substring. Ambiguity at any level is an
    error, never a silent pick.
    """
    cols: List[str] = list(available)

    # Level 1 - exact
    if requested in cols:
        return requested

    # Level 2 - case-insensitive exact
    hits = [c for c in cols if c.lower() == requested.lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ColumnResolutionError(
            f"'{requested}' matches {len(hits)} columns case-insensitively: {hits}"
        )

    # Level 3 - normalised exact (ignores '_', '-', spaces)
    target = _norm(requested)
    hits = [c for c in cols if _norm(c) == target]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ColumnResolutionError(
            f"'{requested}' matches {len(hits)} columns after normalisation: {hits}"
        )

    if allow_fuzzy:
        # Level 4 - unique prefix
        hits = [c for c in cols if _norm(c).startswith(target)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ColumnResolutionError(
                f"'{requested}' is an ambiguous prefix of {hits}. Rename or specify exactly."
            )

        # Level 5 - unique substring
        hits = [c for c in cols if target in _norm(c)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ColumnResolutionError(
                f"'{requested}' is an ambiguous substring of {hits}. Rename or specify exactly."
            )

    if required:
        raise ColumnResolutionError(
            f"Column '{requested}' not found. Available: {sorted(cols)}"
        )
    return None


def resolve_many(
    requested: Sequence[str],
    available: Iterable[str],
    *,
    allow_fuzzy: bool = True,
    required: bool = True,
) -> Dict[str, Optional[str]]:
    """Resolve several names and reject a mapping that is not injective."""
    cols = list(available)
    mapping = {
        r: resolve_column(r, cols, allow_fuzzy=allow_fuzzy, required=required)
        for r in requested
    }
    seen: Dict[str, str] = {}
    for req, got in mapping.items():
        if got is None:
            continue
        if got in seen:
            raise ColumnResolutionError(
                f"'{req}' and '{seen[got]}' both resolved to '{got}'."
            )
        seen[got] = req
    return mapping
