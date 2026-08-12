"""Map an arbitrary dataset's column names onto the names a model declares.

A trained `.keras` model is the authority on what its inputs are called, so the
target list is read off the model rather than hardcoded. This module holds the
pure logic (suggest, profile, apply, persist); the interactive prompting lives in
`scripts/map_columns.py` so the mapping can also be replayed non-interactively.

**Renaming only.** Values are copied through untouched — same dtype, same
numbers, NaNs included. Unit conversion, imputation, and categorical recoding
are deliberately out of scope and left to the caller, so the output GeoPackage
differs from the input in column *names* alone.

A mapping is a flat ``{model_column: source_column}`` dict, JSON-serializable,
so a run is reproducible:

    {"Slope_mean": "Slope",
     "BUK_mean":   "Bulk Density",
     "type":       "Soil Type"}
"""

import difflib
import json
import re

import numpy as np
import pandas as pd

GEOMETRY_COL = "geometry"

# Suffixes that carry the zonal-statistic, not the variable. Stripped before
# fuzzy matching so `Slope` scores against `Slope_mean` on the variable alone.
STAT_SUFFIXES = ("_mean", "_max", "_min", "_majority", "_sum", "_std", "_median")

# Below this similarity a name pair is noise rather than a suggestion.
MIN_SUGGESTION_SCORE = 0.35
MAX_SUGGESTIONS = 5

PROFILE_PERCENTILES = (0, 50, 100)


def normalize_name(name):
    """Reduce a column name to its comparable core (case, punctuation, stat)."""
    text = str(name).strip().lower()
    for suffix in STAT_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return re.sub(r"[^a-z0-9]", "", text)


# Similarity floors for the two structural matches that beat edit distance.
SUBSTRING_SCORE = 0.85
ABBREVIATION_SCORE = 0.80

# Structural matches would otherwise all tie at their floor (`Prc_mean` matches
# both `Precipitation` and `Profile Curvature` as an abbreviation), so a
# tie-break is folded back in. Small enough that the two tiers stay disjoint:
# abbreviations land in [0.80, 0.84], substrings in [0.85, 0.89].
TIE_BREAK_WEIGHT = 0.04

# The tie-break blends edit ratio with shared-prefix fraction. Prefix carries
# real weight because these abbreviations are prefix-preserving (`SoilThc` /
# `Soil Thickness`, `Prc` / `Precipitation`), which is what separates a genuine
# abbreviation from a coincidental subsequence (`SIlt` also hides in `SoilThc`).
PREFIX_WEIGHT = 0.5

# Shorter abbreviations produce coincidental subsequence hits ('cat' inside
# 'Precipitation'), so the boost only applies from this length up.
MIN_ABBREVIATION_LEN = 3


def is_abbreviation(short, long):
    """True when every character of ``short`` appears in order within ``long``.

    This is what catches the house naming style — `Prc_mean` for `Precipitation`,
    `BUK_mean` for `Bulk Density`, `SoilThc_mean` for `Soil Thickness` — none of
    which edit distance ranks anywhere near the top.
    """
    if len(short) < MIN_ABBREVIATION_LEN or len(short) > len(long):
        return False
    position = 0
    for char in short:
        position = long.find(char, position)
        if position < 0:
            return False
        position += 1
    return True


def prefix_fraction(short, long):
    """Fraction of ``short`` that ``long`` shares as a leading prefix."""
    shared = 0
    for a_char, b_char in zip(short, long):
        if a_char != b_char:
            break
        shared += 1
    return shared / len(short) if short else 0.0


def tie_break(short, long, ratio):
    """Blend edit ratio and prefix agreement into a [0, 1] ordering nudge."""
    prefix = prefix_fraction(short, long)
    return (1 - PREFIX_WEIGHT) * ratio + PREFIX_WEIGHT * prefix


def similarity(target, candidate):
    """Score a candidate column against a target name in [0, 1].

    Substring containment and abbreviation are scored above raw edit distance
    because the real-world cases are stat-suffix drift (`Slope` vs `Slope_mean`)
    and abbreviation (`Prc_mean` vs `Precipitation`), which edit distance
    under-scores badly.
    """
    a, b = normalize_name(target), normalize_name(candidate)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    short, long = sorted((a, b), key=len)
    nudge = TIE_BREAK_WEIGHT * tie_break(short, long, ratio)

    if a in b or b in a:
        return max(ratio, SUBSTRING_SCORE + nudge)
    if is_abbreviation(short, long):
        return max(ratio, ABBREVIATION_SCORE + nudge)
    return ratio


def suggest_candidates(target, columns, limit=MAX_SUGGESTIONS):
    """Return ``[(column, score)]`` for the best matches, best first."""
    scored = [
        (col, similarity(target, col))
        for col in columns
        if col != GEOMETRY_COL
    ]
    ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
    return [pair for pair in ranked[:limit] if pair[1] >= MIN_SUGGESTION_SCORE]


def column_profile(series):
    """One-line numeric profile, or a value tally for non-numeric columns.

    Returned as a string rather than a dict because its only consumer is the
    operator reading the prompt — a wrong unit is obvious from the numbers.
    """
    values = pd.to_numeric(series, errors="coerce")
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        tally = series.value_counts().head(4).to_dict()
        return f"non-numeric, top values: {tally}"

    lo, mid, hi = np.percentile(finite, PROFILE_PERCENTILES)
    n_nan = int(values.isna().sum())
    n_zero = int((finite == 0).sum())
    return (
        f"min {lo:<12.4g} p50 {mid:<12.4g} max {hi:<12.4g}"
        f"  nan={n_nan} zero={n_zero}"
    )


def _mapped_series(df, target, source):
    """Copy ``source`` out under the name ``target``, values and dtype untouched."""
    if source not in df.columns:
        raise KeyError(
            f"Mapping for '{target}' reads column '{source}', which is not in the "
            f"data. Available: {sorted(c for c in df.columns if c != GEOMETRY_COL)}"
        )
    return df[source].rename(target)


def _drop_case_collisions(gdf, targets):
    """Drop originals that would collide with a target name only by case.

    GeoPackage field names are case-insensitive, so keeping `Landslide` beside a
    mapped `landslide` makes the write fail with a bare
    ``FieldError: Error adding field``. The target is what the model reads, so
    the original loses.
    """
    lowered = {name.lower() for name in targets}
    collisions = [
        col for col in gdf.columns
        if col not in targets and col != GEOMETRY_COL and col.lower() in lowered
    ]
    if collisions:
        print(f"  Dropping case-colliding originals: {collisions}")
    return gdf.drop(columns=collisions)


def apply_mapping(gdf, mapping, keep_extra=False):
    """Return a NEW GeoDataFrame carrying the model's column names.

    Values are copied verbatim — this only renames. By default just the mapped
    columns and the geometry survive, which is deliberate:
    `py_files.data.preprocessing` runs `dropna` across every column it is handed,
    so an unmapped column with NaNs would silently delete rows the model could
    otherwise have scored. Pass ``keep_extra=True`` to retain the originals.
    """
    if not mapping:
        raise ValueError("Mapping is empty — nothing to write.")

    base = (
        _drop_case_collisions(gdf, set(mapping))
        if keep_extra
        else gdf[[GEOMETRY_COL]].copy()
    )
    mapped = {
        target: _mapped_series(gdf, target, source)
        for target, source in mapping.items()
    }

    out = base.assign(**mapped)
    out.set_geometry(GEOMETRY_COL, inplace=True)
    out.crs = gdf.crs
    return out


def save_mapping(mapping, path, metadata=None):
    """Persist a mapping (plus provenance) so the run can be replayed."""
    payload = {"metadata": metadata or {}, "columns": mapping}
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def load_mapping(path):
    """Load a mapping written by `save_mapping`, validating its shape.

    Accepts the flat ``{"target": "source"}`` form. A ``{"from": "source"}``
    object is also read, so mappings written before this was rename-only still
    replay; any transform keys they carry are ignored rather than half-applied.
    """
    with open(path) as handle:
        payload = json.load(handle)

    columns = payload.get("columns")
    if not isinstance(columns, dict) or not columns:
        raise ValueError(f"{path} has no non-empty 'columns' object.")

    resolved = {}
    for target, spec in columns.items():
        if isinstance(spec, str):
            resolved[target] = spec
            continue
        if isinstance(spec, dict) and isinstance(spec.get("from"), str):
            ignored = sorted(set(spec) - {"from"})
            if ignored:
                print(
                    f"  '{target}': ignoring {ignored} — this script renames only, "
                    "it does not transform values."
                )
            resolved[target] = spec["from"]
            continue
        raise ValueError(
            f"Mapping entry for '{target}' must be a source column name, got {spec!r}."
        )
    return resolved
