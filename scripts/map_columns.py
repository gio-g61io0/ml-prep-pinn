"""Interactively rename a dataset's columns to the names a model expects.

**Renaming only.** Values pass through untouched — same dtype, same numbers,
NaNs included. Unit conversion, imputation, and categorical recoding are out of
scope by design; do those separately, before or after this step. The output
GeoPackage differs from the input in column *names* alone.

The model's `.keras` file is the source of truth for the target names, so this
works for any variant without editing the script. For each model input it ranks
the dataset's columns by name similarity, shows both distributions side by side,
and asks which one to use — then writes a new GeoPackage carrying the model
names, plus a JSON mapping so the same choices can be replayed later.

    # Interactive — pick a source column for each model input
    venv/bin/python -m scripts.map_columns \
        --data datasets/SlopeUnit_Alldata_CIanjurIndo_EQ2022.gpkg \
        --model outputs/model/production-model-v3-1.keras \
        --output datasets/cianjur_mapped.gpkg \
        --reference ~/Documents/ml-prep/ML-PREP-2025/learn/data/SU_17_training_v3_contri.gpkg

    # Replay a saved mapping, no prompts
    venv/bin/python -m scripts.map_columns --data ... --model ... --output ... \
        --replay datasets/cianjur_mapped.mapping.json

The `--reference` dataset is what training used. It is optional but strongly
recommended: printing the training distribution next to the candidate's is how a
unit mismatch (kN/m³ vs cg/cm³, g vs %g) becomes visible — this script will not
fix one, so seeing it here is how you learn which conversions you still owe.
"""

import argparse
import os
import sys

import geopandas as gpd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_files.column_mapping import (  # noqa: E402
    GEOMETRY_COL,
    apply_mapping,
    column_profile,
    load_mapping,
    save_mapping,
    suggest_candidates,
)
from run_model_v3 import LABEL_COL, load_trained_model  # noqa: E402

MAPPING_SUFFIX = ".mapping.json"
SKIP_TOKEN = "s"
LIST_TOKEN = "?"
STRING_DTYPES = ("string", tf.string)

# Some model inputs are computed by the inference script rather than read from
# the file: `run_model_v3.load_and_preprocess` calls `add_soil_texture_index`,
# which needs the three texture fractions. Prompting for `soil_texture_idx`
# itself would be a dead end, so its upstream columns are mapped instead.
DERIVED_INPUTS = {
    "soil_texture_idx": ("Clay_mean", "Silt_mean", "Sand_mean"),
}


def model_targets(model, extra=()):
    """Return ``[(column_name, is_string)]`` for everything the mapping must fill.

    Model inputs listed in `DERIVED_INPUTS` are replaced by the columns they are
    computed from, and ``extra`` appends any further columns the caller's
    preprocessing needs. Order is preserved and duplicates are dropped.
    """
    targets, seen = [], set()

    def add(name, is_string):
        if name not in seen:
            seen.add(name)
            targets.append((name, is_string))

    for tensor in model.inputs:
        is_string = tensor.dtype in STRING_DTYPES
        if tensor.name in DERIVED_INPUTS:
            print(
                f"  '{tensor.name}' is derived during preprocessing — mapping its "
                f"sources instead: {list(DERIVED_INPUTS[tensor.name])}"
            )
            for source in DERIVED_INPUTS[tensor.name]:
                add(source, False)
            continue
        add(tensor.name, is_string)

    for name in extra:
        add(name, False)
    return targets


def string_vocabulary(model):
    """Return the StringLookup vocabulary, or None when it can't be pinned down.

    Only unambiguous cases are reported: several lookup layers would make the
    input-to-vocabulary pairing a guess, and a wrong vocabulary silently maps
    every row to [UNK].
    """
    lookups = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.StringLookup)
    ]
    if len(lookups) != 1:
        return None
    return [str(v) for v in lookups[0].get_vocabulary()]


def print_target_header(index, total, target, is_string, reference, vocabulary):
    """Announce the model input being mapped, with its training distribution."""
    kind = "string" if is_string else "numeric"
    print(f"\n[{index}/{total}] Model input: {target}  ({kind})")

    if reference is not None and target in reference.columns:
        print(f"    training data : {column_profile(reference[target])}")
    elif reference is not None:
        print(f"    training data : '{target}' not in the reference dataset")

    if is_string and vocabulary:
        # Shown, not enforced: recoding source codes into this vocabulary is a
        # value change, so it stays the caller's job. Anything outside the list
        # lands on [UNK] at inference.
        print(f"    model vocabulary: {vocabulary}  (recode to these yourself)")


def print_candidates(candidates, df):
    """List the ranked source columns with their distributions."""
    if not candidates:
        print("    No similarly-named column found.")
        return
    print("    Candidates:")
    for number, (column, score) in enumerate(candidates, start=1):
        print(f"      {number}) {column:<24} {column_profile(df[column])}  [match {score:.2f}]")


def read_choice(candidates, columns):
    """Resolve the operator's answer to a source column, or ``SKIP_TOKEN``."""
    prompt = (
        f"    Pick 1-{len(candidates)}, or type a column name, "
        f"'{SKIP_TOKEN}'=skip, '{LIST_TOKEN}'=list all: "
    )
    while True:
        answer = input(prompt).strip()
        if answer == SKIP_TOKEN:
            return answer
        if answer == LIST_TOKEN:
            print(f"    All columns: {sorted(c for c in columns if c != GEOMETRY_COL)}")
            continue
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            return candidates[int(answer) - 1][0]
        if answer in columns:
            return answer
        print(f"    '{answer}' is not a column in this dataset. Try again.")


def prompt_for_target(df, target, is_string, reference, vocabulary, index, total):
    """Ask which source column becomes ``target``; None means skip."""
    print_target_header(index, total, target, is_string, reference, vocabulary)
    candidates = suggest_candidates(target, list(df.columns))
    print_candidates(candidates, df)

    choice = read_choice(candidates, list(df.columns))
    if choice == SKIP_TOKEN:
        print(f"    Skipped '{target}'.")
        return None
    print(f"    '{choice}' -> '{target}' (values unchanged)")
    return choice


def build_mapping_interactively(df, targets, reference, vocabulary):
    """Walk every model input, returning the mapping the operator chose."""
    mapping = {}
    for index, (target, is_string) in enumerate(targets, start=1):
        source = prompt_for_target(
            df, target, is_string, reference, vocabulary, index, len(targets),
        )
        if source is not None:
            mapping[target] = source
    return mapping


def map_label_column(df, mapping):
    """Offer to carry the inventory label across, since it drives every metric."""
    if LABEL_COL in mapping or LABEL_COL in df.columns:
        return mapping

    candidates = suggest_candidates(LABEL_COL, list(df.columns))
    if not candidates:
        return mapping

    print(f"\nLabel column '{LABEL_COL}' is absent — needed for AUC / confusion metrics.")
    print_candidates(candidates, df)
    choice = read_choice(candidates, list(df.columns))
    if choice == SKIP_TOKEN:
        return mapping
    return {**mapping, LABEL_COL: choice}


def report_result(mapping, targets, out_gdf, output_path, mapping_path):
    """Summarize what was written and, loudly, what was not."""
    missing = [name for name, _ in targets if name not in mapping]

    print(f"\nWrote {len(out_gdf)} features to {output_path}")
    print(f"Wrote mapping to  {mapping_path}")
    print(f"Renamed {len(mapping)} columns (values unchanged): {sorted(mapping)}")
    if missing:
        print(
            f"\n  UNMAPPED model inputs: {missing}\n"
            "  Inference will fail until these columns exist. Re-run and map them,\n"
            "  or add them to the data yourself before inference."
        )
    print(
        "\n  Reminder: values were copied verbatim. Any unit conversion, imputation,\n"
        "  or categorical recoding the model needs is still yours to apply."
    )


def run(data_path, model_path, output_path, mapping_path=None, replay_path=None,
        reference_path=None, keep_extra=False, layer=None, extra_targets=()):
    print(f"Loading data  from {data_path}")
    gdf = gpd.read_file(data_path, layer=layer)
    print(f"  {len(gdf)} features, {len(gdf.columns) - 1} attribute columns, CRS {gdf.crs}")

    print(f"Loading model from {model_path}")
    model = load_trained_model(model_path)
    print(f"  Model declares {len(model.inputs)} inputs: {[t.name for t in model.inputs]}")
    targets = model_targets(model, extra=extra_targets)
    vocabulary = string_vocabulary(model)
    print(f"  Columns to map ({len(targets)}): {[name for name, _ in targets]}")

    reference = None
    if reference_path:
        print(f"Loading reference (training) data from {reference_path}")
        reference = gpd.read_file(os.path.expanduser(reference_path))

    if replay_path:
        print(f"Replaying mapping from {replay_path}")
        mapping = load_mapping(replay_path)
    else:
        mapping = build_mapping_interactively(gdf, targets, reference, vocabulary)
        mapping = map_label_column(gdf, mapping)

    out_gdf = apply_mapping(gdf, mapping, keep_extra=keep_extra)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_gdf.to_file(output_path, driver="GPKG")

    mapping_path = mapping_path or os.path.splitext(output_path)[0] + MAPPING_SUFFIX
    save_mapping(mapping, mapping_path, metadata={
        "source": data_path,
        "model": model_path,
        "reference": reference_path,
        "keep_extra": keep_extra,
    })

    report_result(mapping, targets, out_gdf, output_path, mapping_path)
    return out_gdf, mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map a dataset's column names onto a model's input names.",
    )
    parser.add_argument("--data", required=True, help="Source GeoPackage")
    parser.add_argument("--model", required=True, help=".keras model defining the target names")
    parser.add_argument("--output", required=True, help="GeoPackage to write with mapped names")
    parser.add_argument("--layer", default=None, help="Source layer name (default: first layer)")
    parser.add_argument(
        "--replay", default=None,
        help="Apply a saved mapping JSON instead of prompting",
    )
    parser.add_argument(
        "--mapping-out", default=None,
        help=f"Where to save the mapping JSON (default: <output>{MAPPING_SUFFIX})",
    )
    parser.add_argument(
        "--reference", default=None,
        help="Training GeoPackage; its distributions are printed beside each candidate "
             "so unit mismatches are visible while mapping",
    )
    parser.add_argument(
        "--extra-targets", default="",
        help="Comma-separated extra column names to map beyond the model's inputs "
             "(for features your preprocessing derives or needs)",
    )
    parser.add_argument(
        "--keep-extra", action="store_true",
        help="Keep the unmapped original columns too. Off by default because "
             "preprocessing() dropna's across all columns and would delete rows",
    )
    args = parser.parse_args()

    run(
        args.data, args.model, args.output,
        mapping_path=args.mapping_out,
        replay_path=args.replay,
        reference_path=args.reference,
        keep_extra=args.keep_extra,
        layer=args.layer,
        extra_targets=[c.strip() for c in args.extra_targets.split(",") if c.strip()],
    )
