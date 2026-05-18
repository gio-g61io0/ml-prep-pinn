"""Two-layer feature selection (GA + Elastic Net) adapted from
Amini & Hu, 2020 — arXiv:2001.11177.

Self-contained: depends only on numpy, pandas, scikit-learn. No TensorFlow.

Adaptations from the paper:
- Binary classification target (landslide y/n) instead of regression.
- Inner surrogate is logistic Elastic Net (saga solver) with class_weight='balanced'.
- Fitness uses (1 - mean AUC) in place of relative RMSE.

Mandatory physics features are EXCLUDED from the GA search space and
concatenated back into the final selection unconditionally.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


MANDATORY_PHYSICS_COLS_V3 = [
    "Slope_mean",
    "BUK_mean",
    "Prc_mean",
    "ContributingFactor_mean",
    "SoilThc_mean",
    "soil_texture_idx",
]


# --------------------------------------------------------------------------- #
# Encoding                                                                    #
# --------------------------------------------------------------------------- #

def _encode_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """Standardize numerics, one-hot categoricals.

    Returns
    -------
    X : (n_samples, n_encoded_cols) ndarray
    blocks : list of (feature_name, column_index_array) tuples — one entry per
        original feature, mapping it to its column slice inside X. Used by the
        GA to assemble sub-matrices when a feature's bit is on.
    """
    pieces: list[np.ndarray] = []
    blocks: list[tuple[str, np.ndarray]] = []
    cursor = 0

    if numeric_cols:
        scaler = StandardScaler()
        num_arr = scaler.fit_transform(df[numeric_cols].to_numpy(dtype=float))
        for j, name in enumerate(numeric_cols):
            blocks.append((name, np.array([cursor + j], dtype=int)))
        pieces.append(num_arr)
        cursor += num_arr.shape[1]

    for name in categorical_cols:
        dummies = pd.get_dummies(df[name].astype(str), prefix=name, dummy_na=False)
        arr = dummies.to_numpy(dtype=float)
        idx = np.arange(cursor, cursor + arr.shape[1], dtype=int)
        blocks.append((name, idx))
        pieces.append(arr)
        cursor += arr.shape[1]

    if not pieces:
        raise ValueError("No candidate features supplied to GA-EN selection")

    X = np.concatenate(pieces, axis=1)
    return X, blocks


# --------------------------------------------------------------------------- #
# GA fitness                                                                  #
# --------------------------------------------------------------------------- #

def _select_columns(b: np.ndarray, blocks: list[tuple[str, np.ndarray]]) -> np.ndarray:
    on = [blocks[i][1] for i in range(len(b)) if b[i] == 1]
    if not on:
        return np.empty((0,), dtype=int)
    return np.concatenate(on)


def _evaluate_individual(
    b: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    blocks: list[tuple[str, np.ndarray]],
    w_auc: float,
    w_size: float,
    cv_folds: int,
    random_state: int,
) -> float:
    """Fitness: minimize (w_auc * (1 - AUC) + w_size * fraction_of_features).

    Saga can emit numpy RuntimeWarnings (overflow / divide-by-zero / invalid)
    on small or ill-conditioned subsets — these are convergence noise, not
    correctness bugs. We suppress them inside the fit and detect a NaN AUC
    afterward so a poorly-converged fold scores as random (0.5) rather than
    tainting the population.
    """
    n_features = len(b)
    n_selected = int(b.sum())
    size_term = n_selected / n_features

    if n_selected == 0:
        return float(w_auc * 1.0 + w_size * 0.0 + 1.0)  # heavy penalty

    cols = _select_columns(b, blocks)
    X_sub = X[:, cols]

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    aucs: list[float] = []
    with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        for tr_idx, va_idx in skf.split(X_sub, y):
            clf = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                tol=1e-3,
                random_state=random_state,
            )
            try:
                clf.fit(X_sub[tr_idx], y[tr_idx])
                proba = clf.predict_proba(X_sub[va_idx])[:, 1]
                if np.any(~np.isfinite(proba)):
                    aucs.append(0.5)
                    continue
                auc = roc_auc_score(y[va_idx], proba)
                aucs.append(0.5 if not np.isfinite(auc) else float(auc))
            except (ValueError, FloatingPointError):
                aucs.append(0.5)

    err_term = 1.0 - float(np.mean(aucs))
    return float(w_auc * err_term + w_size * size_term)


# --------------------------------------------------------------------------- #
# GA core                                                                     #
# --------------------------------------------------------------------------- #

def _random_individual(rng: np.random.Generator, n_features: int) -> np.ndarray:
    b = rng.integers(0, 2, size=n_features, dtype=np.int8)
    if b.sum() == 0:
        b[rng.integers(0, n_features)] = 1
    return b


def _single_point_crossover(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = len(p1)
    if n < 2:
        return p1.copy(), p2.copy()
    cut = int(rng.integers(1, n))
    c1 = np.concatenate([p1[:cut], p2[cut:]])
    c2 = np.concatenate([p2[:cut], p1[cut:]])
    return c1, c2


def _mutate(b: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    flips = rng.random(b.shape) < rate
    out = b.copy()
    out[flips] ^= 1
    if out.sum() == 0:  # never return all-zero
        out[rng.integers(0, len(out))] = 1
    return out


def _run_single_ga(
    X: np.ndarray,
    y: np.ndarray,
    blocks: list[tuple[str, np.ndarray]],
    pop_size: int,
    n_generations: int,
    elite_size: int,
    random_size: int,
    mutation_rate: float,
    w_auc: float,
    w_size: float,
    cv_folds: int,
    rng: np.random.Generator,
    verbose: bool,
    repeat_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one GA repeat. Returns (final_best_bitstring, trajectory).

    `trajectory` has shape (n_generations, n_features). Row `g` is the
    best individual of generation g (after fitness-sorting), letting
    callers compute per-(feature, generation) presence statistics.
    """
    n_features = len(blocks)
    population = [_random_individual(rng, n_features) for _ in range(pop_size)]
    trajectory = np.zeros((n_generations, n_features), dtype=np.int8)

    for gen in range(n_generations):
        fitnesses = np.array(
            [
                _evaluate_individual(
                    b, X, y, blocks, w_auc, w_size, cv_folds,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
                for b in population
            ]
        )
        order = np.argsort(fitnesses)
        population = [population[i] for i in order]
        fitnesses = fitnesses[order]
        trajectory[gen] = population[0]
        if verbose:
            best = fitnesses[0]
            print(
                f"  [GA repeat {repeat_idx}] gen {gen + 1}/{n_generations}  "
                f"best fitness={best:.4f}  selected={int(population[0].sum())}/{n_features}"
            )

        elites = population[:elite_size]
        randoms = [
            population[int(rng.integers(elite_size, len(population)))]
            for _ in range(random_size)
        ]
        parents = elites + randoms

        # Each parent pair produces children; refill to pop_size
        children: list[np.ndarray] = []
        while len(children) < pop_size:
            i = int(rng.integers(0, len(parents)))
            j = int(rng.integers(0, len(parents)))
            if i == j:
                j = (j + 1) % len(parents)
            c1, c2 = _single_point_crossover(parents[i], parents[j], rng)
            children.append(_mutate(c1, mutation_rate, rng))
            if len(children) < pop_size:
                children.append(_mutate(c2, mutation_rate, rng))

        # Elitism: keep the best individual unchanged
        children[0] = population[0].copy()
        population = children

    # Final evaluation to pick best
    fitnesses = np.array(
        [
            _evaluate_individual(
                b, X, y, blocks, w_auc, w_size, cv_folds,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            for b in population
        ]
    )
    return population[int(np.argmin(fitnesses))], trajectory


# --------------------------------------------------------------------------- #
# Layer 2: tune EN (diagnostic)                                               #
# --------------------------------------------------------------------------- #

def _tune_and_prune_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    blocks: list[tuple[str, np.ndarray]],
    selected_names: list[str],
    l1_ratios: tuple,
    cv_folds: int,
    random_state: int,
    coef_eps: float = 1e-6,
) -> tuple[float, float, list[str], dict[str, float]]:
    """Layer 2 of GA-EN: tune (α, ρ) on the GA-survivor subset, then refit at
    the tuned hyperparameters and drop features whose total |coef| is below
    `coef_eps`. Mirrors the paper's Layer-2 redundancy elimination, adapted
    for a logistic surrogate.

    Returns (alpha, l1_ratio, pruned_names, coef_magnitudes) where
    `pruned_names` is the subset of `selected_names` whose coefficients
    survived L1 shrinkage, and `coef_magnitudes` is a dict mapping every
    `selected_names` entry to its total |coef| (sum across one-hot block
    columns for categoricals).
    """
    # Build the survivor sub-matrix and a parallel mapping of original feature
    # names to LOCAL column indices inside that sub-matrix (so we can attribute
    # logistic-EN coefficients back to feature names — including one-hot blocks
    # that span multiple columns per categorical).
    sub_pieces: list[np.ndarray] = []
    sub_blocks: list[tuple[str, np.ndarray]] = []
    cursor = 0
    name_to_block = {name: idx for name, idx in blocks}
    for name in selected_names:
        idx = name_to_block[name]
        sub_pieces.append(X[:, idx])
        sub_blocks.append((name, np.arange(cursor, cursor + len(idx), dtype=int)))
        cursor += len(idx)
    X_sub = np.concatenate(sub_pieces, axis=1)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_clf = LogisticRegressionCV(
        Cs=10,
        cv=cv,
        penalty="elasticnet",
        solver="saga",
        l1_ratios=list(l1_ratios),
        class_weight="balanced",
        max_iter=3000,
        tol=1e-3,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=random_state,
    )
    with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        cv_clf.fit(X_sub, y)

    best_C = float(cv_clf.C_[0])
    alpha = 1.0 / best_C if best_C > 0 else float("inf")
    l1_ratio = float(cv_clf.l1_ratio_[0])

    # Refit at the tuned (α, ρ) so we can read coefficients back. CV uses
    # average-of-folds models, but a single fit on the full survivor data at
    # the tuned hyperparameters is what dictates which features survive L1.
    final_clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=best_C,
        l1_ratio=l1_ratio,
        class_weight="balanced",
        max_iter=5000,
        tol=1e-4,
        random_state=random_state,
    )
    with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        final_clf.fit(X_sub, y)

    coefs = np.abs(final_clf.coef_.ravel())  # binary classification → 1D
    pruned: list[str] = []
    coef_magnitudes: dict[str, float] = {}
    for name, local_idx in sub_blocks:
        mag = float(coefs[local_idx].sum())
        coef_magnitudes[name] = mag
        if mag > coef_eps:
            pruned.append(name)

    return alpha, l1_ratio, pruned, coef_magnitudes


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def select_features(
    df: pd.DataFrame,
    target: str = "landslide",
    mandatory_cols: list[str] | None = None,
    candidate_numeric: list[str] | None = None,
    candidate_categorical: list[str] | None = None,
    *,
    fsp: float = 0.5,
    n_ga_repeats: int = 5,
    pop_size: int = 50,
    n_generations: int = 10,
    elite_size: int = 19,
    random_size: int = 1,
    mutation_rate: float = 0.05,
    w_auc: float = 0.8,
    w_size: float = 0.2,
    en_l1_ratios: tuple = (0.1, 0.3, 0.5, 0.7, 0.9),
    cv_folds: int = 3,
    coef_eps: float = 1e-6,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Run GA-EN feature selection.

    Parameters
    ----------
    df : DataFrame containing target + all candidate features.
    target : column name of the binary label (default 'landslide').
    mandatory_cols : columns that MUST appear in the final subset (excluded
        from the GA search). Splits internally into numeric/categorical based
        on dtype.
    candidate_numeric / candidate_categorical : columns the GA may choose from.
        Mandatory cols are removed automatically if mistakenly included.

    Returns
    -------
    dict with keys:
        'numerical'        : list[str]   — mandatory numeric + Layer-2-pruned numeric
        'categorical'      : list[str]   — mandatory categorical + Layer-2-pruned categorical
        'frequencies'      : dict[str, float] — fraction of GA repeats each candidate survived in
        'ga_survivors'     : list[str]   — Layer-1 (GA + FSP) survivors, BEFORE Layer-2 pruning
        'pruned_by_layer2' : list[str]   — features dropped by Layer-2 EN L1 shrinkage
        'en_alpha'         : float       — Layer 2 tuned alpha
        'en_l1_ratio'      : float       — Layer 2 tuned l1_ratio
    """
    mandatory_cols = list(mandatory_cols or [])
    candidate_numeric = [c for c in (candidate_numeric or []) if c not in mandatory_cols]
    candidate_categorical = [c for c in (candidate_categorical or []) if c not in mandatory_cols]

    if target not in df.columns:
        raise ValueError(f"target '{target}' not in df.columns")
    y = df[target].astype(int).to_numpy()
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("target must be binary (0/1)")

    candidates = candidate_numeric + candidate_categorical
    if not candidates:
        raise ValueError("No candidate features supplied to GA-EN selection")

    missing = [c for c in candidates + mandatory_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    X, blocks = _encode_features(df, candidate_numeric, candidate_categorical)
    n_features = len(blocks)
    rng = np.random.default_rng(random_state)

    if verbose:
        print(
            f"GA-EN: {n_features} candidate features "
            f"({len(candidate_numeric)} numeric + {len(candidate_categorical)} categorical), "
            f"{len(mandatory_cols)} mandatory features (held out)."
        )

    # Layer 1: GA repeated n_ga_repeats times
    counts = np.zeros(n_features, dtype=int)
    repeat_bits: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []
    for r in range(n_ga_repeats):
        if verbose:
            print(f"--- GA repeat {r + 1}/{n_ga_repeats} ---")
        # Use a fresh RNG seeded per-repeat so runs are reproducible & independent
        repeat_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        best, trajectory = _run_single_ga(
            X=X, y=y, blocks=blocks,
            pop_size=pop_size, n_generations=n_generations,
            elite_size=elite_size, random_size=random_size,
            mutation_rate=mutation_rate,
            w_auc=w_auc, w_size=w_size,
            cv_folds=cv_folds, rng=repeat_rng,
            verbose=verbose, repeat_idx=r + 1,
        )
        counts += best.astype(int)
        repeat_bits.append(best.astype(int).copy())
        trajectories.append(trajectory)

    feature_names = [blocks[i][0] for i in range(n_features)]
    repeat_selections = pd.DataFrame(
        np.stack(repeat_bits, axis=1),
        index=feature_names,
        columns=[f"repeat_{r + 1}" for r in range(n_ga_repeats)],
    )
    # Per-(feature, repeat) stability score: fraction of generations in which
    # the feature was in the best individual of that repeat. Shape (F, R).
    trajectory_stack = np.stack(trajectories, axis=0)  # (R, G, F)
    trajectory_scores = pd.DataFrame(
        trajectory_stack.mean(axis=1).T,  # (F, R)
        index=feature_names,
        columns=[f"repeat_{r + 1}" for r in range(n_ga_repeats)],
    )
    frequencies = {blocks[i][0]: float(counts[i] / n_ga_repeats) for i in range(n_features)}
    selected_candidates = [name for name, freq in frequencies.items() if freq >= fsp]

    if verbose:
        print(f"Frequencies (FSP={fsp}):")
        for name, freq in sorted(frequencies.items(), key=lambda kv: -kv[1]):
            mark = "*" if freq >= fsp else " "
            print(f"  {mark} {name:30s} {freq:.2f}")

    # Layer 2: tune (α, ρ) AND prune residual redundant features whose
    # logistic-EN coefficient shrinks to zero under L1.
    ga_survivors = list(selected_candidates)
    l2_coefficients: dict[str, float] = {}
    if selected_candidates:
        try:
            en_alpha, en_l1_ratio, layer2_kept, l2_coefficients = _tune_and_prune_elastic_net(
                X=X, y=y, blocks=blocks,
                selected_names=selected_candidates,
                l1_ratios=en_l1_ratios,
                cv_folds=cv_folds,
                random_state=random_state,
                coef_eps=coef_eps,
            )
        except Exception as exc:
            if verbose:
                print(f"Layer-2 tune+prune failed ({exc}); keeping GA survivors as-is")
            en_alpha, en_l1_ratio = float("nan"), float("nan")
            layer2_kept = list(selected_candidates)
    else:
        en_alpha, en_l1_ratio = float("nan"), float("nan")
        layer2_kept = []

    pruned_by_layer2 = [c for c in ga_survivors if c not in layer2_kept]
    if verbose:
        print(f"\nLayer 2 — tuned (alpha={en_alpha:.4g}, l1_ratio={en_l1_ratio:.2f})")
        if pruned_by_layer2:
            print(f"  pruned by L1 shrinkage: {pruned_by_layer2}")
        else:
            print("  no further features pruned by L1")

    # Split Layer-2 survivors back into numeric/categorical, preserving original order
    selected_numeric = [c for c in candidate_numeric if c in layer2_kept]
    selected_cat = [c for c in candidate_categorical if c in layer2_kept]

    mandatory_numeric = [c for c in mandatory_cols if c not in candidate_categorical]
    mandatory_cat = [c for c in mandatory_cols if c in candidate_categorical]

    final_numeric = mandatory_numeric + selected_numeric
    final_cat = mandatory_cat + selected_cat

    result = {
        "numerical": final_numeric,
        "categorical": final_cat,
        "frequencies": frequencies,
        "repeat_selections": repeat_selections,
        "trajectory_scores": trajectory_scores,
        "trajectories": {
            f"repeat_{r + 1}": pd.DataFrame(
                trajectories[r],
                index=[f"gen_{g + 1}" for g in range(n_generations)],
                columns=feature_names,
            )
            for r in range(n_ga_repeats)
        },
        "ga_survivors": ga_survivors,
        "pruned_by_layer2": pruned_by_layer2,
        "l2_coefficients": l2_coefficients,
        "en_alpha": en_alpha,
        "en_l1_ratio": en_l1_ratio,
    }

    assert set(mandatory_cols).issubset(set(result["numerical"]) | set(result["categorical"])), (
        "Mandatory features missing from selected subset"
    )
    return result


# --------------------------------------------------------------------------- #
# Analytics: per-feature decision report                                      #
# --------------------------------------------------------------------------- #

def feature_selection_report(
    df: pd.DataFrame,
    target: str,
    result: dict,
    mandatory_cols: list[str],
    candidate_numeric: list[str],
    candidate_categorical: list[str],
) -> pd.DataFrame:
    """Build a per-feature decision table explaining the selection.

    Columns:
        feature, role, ga_frequency, passed_fsp, l2_coef_magnitude, passed_l2,
        decision, mean_pos, mean_neg, mean_abs_diff, cohens_d, mwu_p, mi_score

    `role` ∈ {mandatory, candidate_numeric, candidate_categorical}.
    `decision` ∈ {KEPT_MANDATORY, KEPT_BY_GA_AND_L2, DROPPED_AT_L1, DROPPED_AT_L2}.

    Statistics are computed from the supplied `df` (drop NaNs first if needed).
    For categorical features only the GA/L2 columns are populated; statistical
    columns are NaN (a categorical variable doesn't have a meaningful mean).
    """
    from scipy.stats import mannwhitneyu
    from sklearn.feature_selection import mutual_info_classif

    y = df[target].astype(int).to_numpy()
    pos_mask = y == 1
    neg_mask = y == 0

    frequencies = result.get("frequencies", {})
    ga_survivors = set(result.get("ga_survivors", []))
    pruned_l2 = set(result.get("pruned_by_layer2", []))
    l2_coefs = result.get("l2_coefficients", {})

    # Mutual info — model-free signal estimate for the same candidates the GA saw
    mi_scores: dict[str, float] = {}
    if candidate_numeric:
        try:
            X_num = df[candidate_numeric].to_numpy(dtype=float)
            mi_num = mutual_info_classif(X_num, y, discrete_features=False, random_state=42)
            for name, score in zip(candidate_numeric, mi_num):
                mi_scores[name] = float(score)
        except Exception:
            for name in candidate_numeric:
                mi_scores[name] = float("nan")
    for name in candidate_categorical:
        try:
            x_cat = pd.Categorical(df[name]).codes.reshape(-1, 1)
            mi_scores[name] = float(
                mutual_info_classif(x_cat, y, discrete_features=True, random_state=42)[0]
            )
        except Exception:
            mi_scores[name] = float("nan")

    rows = []
    all_features = mandatory_cols + candidate_numeric + candidate_categorical
    seen = set()
    for name in all_features:
        if name in seen:
            continue
        seen.add(name)

        if name in mandatory_cols:
            role = "mandatory"
        elif name in candidate_categorical:
            role = "candidate_categorical"
        else:
            role = "candidate_numeric"

        ga_freq = frequencies.get(name, float("nan"))
        passed_fsp = name in ga_survivors if name not in mandatory_cols else None
        l2_mag = l2_coefs.get(name, float("nan"))
        passed_l2 = (
            None if name in mandatory_cols
            else (name in ga_survivors and name not in pruned_l2)
        )

        if name in mandatory_cols:
            decision = "KEPT_MANDATORY"
        elif name not in ga_survivors:
            decision = "DROPPED_AT_L1"
        elif name in pruned_l2:
            decision = "DROPPED_AT_L2"
        else:
            decision = "KEPT_BY_GA_AND_L2"

        # Per-class statistics — meaningful only for numeric or mandatory numeric
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            x = df[name].to_numpy(dtype=float)
            x_pos = x[pos_mask]
            x_neg = x[neg_mask]
            mean_pos = float(np.mean(x_pos)) if len(x_pos) else float("nan")
            mean_neg = float(np.mean(x_neg)) if len(x_neg) else float("nan")
            std_pooled = float(np.sqrt(0.5 * (np.var(x_pos, ddof=1) + np.var(x_neg, ddof=1))))
            cohens_d = (mean_pos - mean_neg) / std_pooled if std_pooled > 0 else float("nan")
            try:
                _, mwu_p = mannwhitneyu(x_pos, x_neg, alternative="two-sided")
            except ValueError:
                mwu_p = float("nan")
        else:
            mean_pos = mean_neg = cohens_d = mwu_p = float("nan")

        rows.append({
            "feature": name,
            "role": role,
            "ga_frequency": ga_freq,
            "passed_fsp": passed_fsp,
            "l2_coef_magnitude": l2_mag,
            "passed_l2": passed_l2,
            "decision": decision,
            "mean_pos": mean_pos,
            "mean_neg": mean_neg,
            "mean_abs_diff": abs(mean_pos - mean_neg) if np.isfinite(mean_pos) and np.isfinite(mean_neg) else float("nan"),
            "cohens_d": cohens_d,
            "mwu_p": mwu_p,
            "mi_score": mi_scores.get(name, float("nan")),
        })

    report_df = pd.DataFrame(rows)
    # Sort: mandatory first, then by decision priority and decreasing strength
    decision_order = {
        "KEPT_MANDATORY": 0,
        "KEPT_BY_GA_AND_L2": 1,
        "DROPPED_AT_L2": 2,
        "DROPPED_AT_L1": 3,
    }
    report_df["_decision_rank"] = report_df["decision"].map(decision_order)
    report_df = report_df.sort_values(
        by=["_decision_rank", "ga_frequency", "l2_coef_magnitude"],
        ascending=[True, False, False],
    ).drop(columns="_decision_rank").reset_index(drop=True)
    return report_df


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #

def plot_ga_frequencies(
    result: dict,
    mandatory_cols: list[str] | None = None,
    fsp: float = 0.5,
    figsize: tuple = (10, 8),
    ax=None,
):
    """Horizontal bar chart of GA per-feature frequency across repeats.

    Mandatory features are omitted (they bypass the GA). Bars above the FSP
    line are colored green (kept by Layer 1), below are red (dropped).
    """
    import matplotlib.pyplot as plt

    mandatory_cols = set(mandatory_cols or [])
    frequencies = result.get("frequencies", {})
    items = [(name, freq) for name, freq in frequencies.items() if name not in mandatory_cols]
    if not items:
        raise ValueError("No non-mandatory features found in result['frequencies']")
    items.sort(key=lambda kv: kv[1], reverse=True)
    names = [n for n, _ in items]
    freqs = [f for _, f in items]
    colors = ["#2ca02c" if f >= fsp else "#d62728" for f in freqs]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.barh(names, freqs, color=colors)
    ax.axvline(fsp, color="black", linestyle="--", linewidth=1, label=f"FSP = {fsp}")
    ax.set_xlabel("Fraction of GA repeats survived")
    ax.set_title("Layer 1: GA feature-selection frequencies")
    ax.set_xlim(0, 1.02)
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    return ax


def plot_repeat_selections(
    result: dict,
    mandatory_cols: list[str] | None = None,
    sort_by_total: bool = True,
    annotate_total: bool = True,
    figsize: tuple = (8, 10),
    ax=None,
):
    """Heatmap of which features were in the best individual of each GA repeat.

    Rows are non-mandatory candidate features; columns are repeats. A filled
    cell means the feature was in that repeat's final-best bitstring. With
    `sort_by_total=True`, rows are ordered by total times selected (most
    consistent at top). With `annotate_total=True`, each row shows its count
    on the right (e.g., "3/5").
    """
    import matplotlib.pyplot as plt

    mandatory_cols = set(mandatory_cols or [])
    sel = result.get("repeat_selections")
    if sel is None:
        raise ValueError(
            "result is missing 'repeat_selections' — re-run select_features() "
            "after the latest module update."
        )

    drop = [c for c in mandatory_cols if c in sel.index]
    if drop:
        sel = sel.drop(index=drop)
    if sel.empty:
        raise ValueError("No non-mandatory features available to plot.")

    totals = sel.sum(axis=1)
    if sort_by_total:
        sel = sel.loc[totals.sort_values(ascending=False, kind="stable").index]
        totals = totals.loc[sel.index]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    n_repeats = sel.shape[1]
    ax.imshow(sel.to_numpy(), aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(n_repeats))
    ax.set_xticklabels(sel.columns, rotation=0)
    ax.set_yticks(range(sel.shape[0]))
    ax.set_yticklabels(sel.index)
    ax.set_xlabel("GA repeat")
    ax.set_title("Per-repeat selection (filled = in repeat's best individual)")

    # Light cell borders so the grid is readable when many features are 0
    ax.set_xticks(np.arange(-0.5, n_repeats, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, sel.shape[0], 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    if annotate_total:
        for i, total in enumerate(totals):
            ax.text(
                n_repeats - 0.4, i, f"  {int(total)}/{n_repeats}",
                va="center", ha="left", fontsize=8, clip_on=False,
            )

    return ax


def plot_trajectory_heatmap(
    result: dict,
    mandatory_cols: list[str] | None = None,
    sort_by_mean: bool = True,
    annotate: bool = True,
    cmap: str = "viridis",
    figsize: tuple = (8, 10),
    ax=None,
):
    """Heatmap of fraction of generations each feature was in the best individual.

    Rows are non-mandatory candidate features, columns are GA repeats, cell
    value ∈ [0, 1] = (# generations feature was in repeat's best individual)
    / n_generations. Continuous shading reveals stability the binary
    final-winner view (`plot_repeat_selections`) misses — a feature that
    only entered the best individual in the last generation has score ~0.1
    even if it ended up in the final winner.
    """
    import matplotlib.pyplot as plt

    mandatory_cols = set(mandatory_cols or [])
    scores = result.get("trajectory_scores")
    if scores is None:
        raise ValueError(
            "result is missing 'trajectory_scores' — re-run select_features() "
            "after the latest module update."
        )

    drop = [c for c in mandatory_cols if c in scores.index]
    if drop:
        scores = scores.drop(index=drop)
    if scores.empty:
        raise ValueError("No non-mandatory features available to plot.")

    if sort_by_mean:
        means = scores.mean(axis=1)
        scores = scores.loc[means.sort_values(ascending=False, kind="stable").index]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(scores.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(scores.shape[1]))
    ax.set_xticklabels(scores.columns, rotation=0)
    ax.set_yticks(range(scores.shape[0]))
    ax.set_yticklabels(scores.index)
    ax.set_xlabel("GA repeat")
    ax.set_title("Generations in best individual (fraction across all generations)")
    plt.colorbar(im, ax=ax, label="Generations selected / total generations")

    if annotate:
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                v = float(scores.iat[i, j])
                color = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    return ax


def plot_l2_coefficients(
    result: dict,
    mandatory_cols: list[str] | None = None,
    include_pruned: bool = False,
    figsize: tuple = (10, 6),
    ax=None,
):
    """Horizontal bar chart of |coef| for Layer-2 features.

    By default plots only survivors (features the L1 shrinkage did NOT zero
    out). Pass `include_pruned=True` to also show the dropped features (they
    will sit at ~0) as a visual sanity check that L1 actually drove them
    below `coef_eps`.
    """
    import matplotlib.pyplot as plt

    mandatory_cols = set(mandatory_cols or [])
    coefs = result.get("l2_coefficients", {})
    pruned = set(result.get("pruned_by_layer2", []))
    if not coefs:
        raise ValueError("No Layer-2 coefficients found in result['l2_coefficients']")

    items = [
        (name, mag) for name, mag in coefs.items()
        if name not in mandatory_cols and (include_pruned or name not in pruned)
    ]
    if not items:
        raise ValueError("No features to plot (all are mandatory or pruned).")
    items.sort(key=lambda kv: kv[1], reverse=True)

    names = [n for n, _ in items]
    mags = [m for _, m in items]
    colors = ["#1f77b4" if n not in pruned else "#d62728" for n in names]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.barh(names, mags, color=colors)
    ax.set_xlabel("|coef| (summed across one-hot columns for categoricals)")
    title = "Layer 2: Elastic Net coefficient magnitudes"
    if include_pruned:
        title += " (blue = survivor, red = pruned by L1)"
    else:
        title += " — survivors only"
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    return ax
