# ------------------------
# IMPORTS
# ------------------------
import numpy as np
import pandas as pd
from numpy.random import default_rng

rng = default_rng(42)


# ------------------------
# RESTRICTED CUBIC SPLINE BASIS (Harrell's formula, Regression Modeling Strategies eq. 2.26)
# ------------------------
def rcs_basis(x, knots=None, nk=4, standardize=True):
    """
    Restricted cubic spline basis for a continuous vector x.
    Returns an (n, nk-1) matrix: column 0 is x itself (linear term),
    columns 1..(nk-2) are the nonlinear spline terms.
    knots: explicit knot locations, or None to use default percentiles.

    standardize: z-score x before building the cubic terms. Cubic terms on
    raw large-scale variables (e.g. production values in the thousands)
    blow up to very different magnitudes than other columns and make the
    design matrix ill-conditioned -> lstsq/SVD failures downstream.
    Standardizing keeps everything comparable; it does not change R^2
    since OLS fit is invariant to linear rescaling of a predictor.
    """
    x = np.asarray(x, dtype=float)

    if np.isinf(x).any():
        raise ValueError("rcs_basis: Inf found in a continuous predictor.")

    # NaNs are allowed through deliberately: arithmetic on a NaN entry
    # propagates to NaN in every basis column for that row, and the
    # regression step (predictor_r2) drops such rows on a per-model basis,
    # matching Hmisc::redun's casewise deletion rather than requiring a
    # globally complete dataset.
    if standardize:
        x_std = np.nanstd(x)
        if x_std > 0:
            x = (x - np.nanmean(x)) / x_std

    if knots is None:
        default_pct = {
            3: [0.10, 0.5, 0.90],
            4: [0.05, 0.35, 0.65, 0.95],
            5: [0.05, 0.275, 0.5, 0.725, 0.95],
            6: [0.05, 0.23, 0.41, 0.59, 0.77, 0.95],
            7: [0.025, 0.1833, 0.3417, 0.5, 0.6583, 0.8167, 0.975],
        }
        pct = default_pct.get(nk, np.linspace(0.05, 0.95, nk))
        knots = np.nanquantile(x, pct)
    knots = np.asarray(knots, dtype=float)
    nk = len(knots)

    if nk < 3:
        raise ValueError("RCS requires at least 3 knots")

    n = len(x)
    basis = np.zeros((n, nk - 1))
    basis[:, 0] = x  # linear term

    k1, kn = knots[0], knots[-1]
    denom = kn - k1

    for j in range(nk - 2):
        kj = knots[j]
        # X_{j+1} = (x-kj)_+^3 - (x-k_{nk-1})_+^3 * (kn - kj)/(kn - k1)
        #                      + (x-k1)_+^3 * (k_{nk-1} - kj)/(kn - k1)
        k_nkm1 = knots[-2]
        term_a = np.clip(x - kj, 0, None) ** 3
        term_b = np.clip(x - kn, 0, None) ** 3 * (kn - kj) / denom
        term_c = np.clip(x - k1, 0, None) ** 3 * (k_nkm1 - kj) / denom
        basis[:, j + 1] = term_a - term_b + term_c

    return basis


# ------------------------
# TERM EXPANSION FOR A DATAFRAME
# expands each continuous predictor into RCS basis columns,
# each categorical predictor into dummy columns.
# returns dict {predictor_name: [column names in the expanded design matrix]}
# and the full expanded DataFrame
# ------------------------
def expand_terms(df, nk=4, min_unique_for_spline=6):
    """
    Expands each predictor into its RCS / dummy basis columns. NaNs are
    preserved (not dropped or imputed here) -- casewise deletion happens
    later, per regression, inside predictor_r2. This mirrors how
    Hmisc::redun reports "Number of NAs" and per-variable missing counts
    up front but still runs each submodel on whatever rows are complete
    for that particular submodel.
    """
    expanded = {}
    col_groups = {}

    missing_per_var = df.isna().sum()
    missing_per_var = missing_per_var[missing_per_var > 0]
    n_incomplete_rows = int(df.isna().any(axis=1).sum())

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) >= min_unique_for_spline:
            basis = rcs_basis(series.values, nk=nk)
            names = [f"{col}_s{i}" for i in range(basis.shape[1])]
            for i, name in enumerate(names):
                expanded[name] = basis[:, i]
            col_groups[col] = names

        else:
            # CATEGORICAL OR LOW-CARDINALITY NUMERIC -> DUMMY EXPANSION
            # dummy_na=False (default) would silently code a missing
            # category as all-zeros across dummies, which is wrong --
            # explicitly re-mark those rows as NaN so they get dropped
            # casewise downstream instead of being treated as the
            # reference level.
            dummies = pd.get_dummies(series, prefix=col, drop_first=True).astype(float)
            is_na = series.isna().values
            dummies.loc[is_na, :] = np.nan
            names = list(dummies.columns)
            for name in names:
                expanded[name] = dummies[name].values
            col_groups[col] = names

    expanded_df = pd.DataFrame(expanded, index=df.index)
    missing_report = {
        "n_incomplete_rows": n_incomplete_rows,
        "per_variable": missing_per_var.to_dict(),
    }
    return expanded_df, col_groups, missing_report


# ------------------------
# CANONICAL VARIATE: reduce a multi-column target block to a single score
# via its first principal component (practical stand-in for the first
# canonical variate referenced in Harrell's algorithm)
# ------------------------
def first_canonical_variate(Y):
    if Y.shape[1] == 1:
        return Y[:, 0]
    Yc = Y - Y.mean(axis=0)
    u, s, vt = np.linalg.svd(Yc, full_matrices=False)
    return u[:, 0] * s[0]


# ------------------------
# OLS R^2 AND ADJUSTED R^2 FOR "TARGET ~ OTHERS"
# ------------------------
def ols_r2(y, X, adjusted=True, ridge=1e-8):
    n, p = X.shape
    Xd = np.column_stack([np.ones(n), X])

    if not np.all(np.isfinite(Xd)) or not np.all(np.isfinite(y)):
        raise ValueError(
            "ols_r2: non-finite values in design matrix or target. "
            "Check for NaN/Inf upstream (expand_terms should already catch NaNs)."
        )

    try:
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    except np.linalg.LinAlgError:
        # FALLBACK FOR ILL-CONDITIONED / NON-CONVERGING SVD:
        # small ridge penalty on the normal equations stabilizes the solve.
        # intercept column left unpenalized.
        XtX = Xd.T @ Xd
        penalty = ridge * np.eye(XtX.shape[0])
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(XtX + penalty, Xd.T @ y)

    yhat = Xd @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if adjusted:
        if n - p - 1 <= 0:
            return r2
        r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return r2


def predictor_r2(target, remaining, col_groups, expanded_df, adjusted=True, min_rows_per_param=5):
    """
    R^2 (or adjusted R^2) predicting `target` from all other predictors in
    `remaining`. Casewise deletion: only rows complete across the target
    block AND the predictor block for THIS specific submodel are used --
    a different submodel (different `remaining` set) may use a different
    subset of rows, exactly as Hmisc::redun does per-regression.
    """
    target_cols = col_groups[target]
    other_cols = [c for name in remaining if name != target for c in col_groups[name]]

    if len(other_cols) == 0:
        return 0.0, 0

    sub = expanded_df[target_cols + other_cols]
    complete = sub.notna().all(axis=1)
    n_complete = int(complete.sum())

    n_params = len(other_cols) + 1
    if n_complete < max(n_params + 2, min_rows_per_param * n_params):
        # not enough complete cases to fit this submodel reliably
        return np.nan, n_complete

    Y = sub.loc[complete, target_cols].values
    X = sub.loc[complete, other_cols].values
    y = first_canonical_variate(Y)

    return ols_r2(y, X, adjusted=adjusted), n_complete


# ------------------------
# MAIN REDUNDANCY ALGORITHM
# mirrors Hmisc::redun's iterative elimination with the "undo" check:
# a variable is only dropped for good if doing so does not cause a
# previously dropped variable to fall below the threshold again.
# ------------------------
def redun_py(df, r2_threshold=0.9, adjusted=True, nk=4, verbose=True):
    expanded_df, col_groups, missing_report = expand_terms(df, nk=nk)
    remaining = list(col_groups.keys())
    removed_order = []  # list of (name, r2_at_removal)

    if verbose:
        print("Redundancy Analysis")
        print(f"n: {len(df)}   p: {len(col_groups)}   nk: {nk}")
        print(f"Number of incomplete rows: {missing_report['n_incomplete_rows']}")
        if missing_report["per_variable"]:
            print("Frequencies of Missing Values Due to Each Variable")
            print(pd.Series(missing_report["per_variable"]).to_string())
        print(f"R2 cutoff: {r2_threshold}   Type: {'adjusted' if adjusted else 'ordinary'}")
        print()

    # INITIAL R2 TABLE (EACH VARIABLE PREDICTED FROM ALL OTHERS), FOR REPORTING
    initial_scores = {}
    for name in remaining:
        r2, n_used = predictor_r2(name, remaining, col_groups, expanded_df, adjusted)
        initial_scores[name] = (r2, n_used)

    if verbose:
        print("R2 with which each variable can be predicted from all others:")
        for name, (r2, n_used) in initial_scores.items():
            r2_str = f"{r2:.3f}" if not np.isnan(r2) else "NA"
            print(f"  {name:>15s}: {r2_str}   (n={n_used})")
        print()

    while True:
        if len(remaining) < 2:
            break

        scores = {}
        for name in remaining:
            r2, n_used = predictor_r2(name, remaining, col_groups, expanded_df, adjusted)
            scores[name] = r2

        # NaN (too few complete cases for this submodel) can't be compared -- skip it
        valid_scores = {k: v for k, v in scores.items() if not np.isnan(v)}
        if not valid_scores:
            break

        worst_name = max(valid_scores, key=valid_scores.get)
        worst_r2 = valid_scores[worst_name]

        if worst_r2 < r2_threshold:
            break

        # TENTATIVELY DROP AND CHECK WHETHER ANY PREVIOUSLY REMOVED
        # VARIABLE WOULD NO LONGER CLEAR THE THRESHOLD FROM THE SMALLER SET
        trial_remaining = [n for n in remaining if n != worst_name]

        broken = False
        for prev_name, _ in removed_order:
            prev_check, _ = predictor_r2(
                prev_name, trial_remaining + [prev_name], col_groups, expanded_df, adjusted
            )
            if np.isnan(prev_check) or prev_check < r2_threshold:
                broken = True
                if verbose:
                    print(
                        f"  -> {worst_name} atilmiyor: {prev_name} artik "
                        f"esik altina duser (R2={prev_check:.3f})"
                    )
                break

        if broken:
            break

        if verbose:
            print(f"Removing {worst_name:>15s}  (R2={worst_r2:.4f})")

        removed_order.append((worst_name, worst_r2))
        remaining.remove(worst_name)

    return {
        "In": remaining,
        "Out": [name for name, _ in removed_order],
        "rsq": dict(removed_order),
        "initial_r2": initial_scores,
        "missing_report": missing_report,
    }


# ------------------------
# CATEGORICAL-SPECIFIC CHECK (strict version):
# a categorical predictor is redundant only if EVERY dummy derived from
# it is individually redundant given the rest
# ------------------------
def categorical_strict_redundant(cat_col, df, remaining_others, nk=4, r2_threshold=0.9, adjusted=True):
    dummies = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=True).astype(float)
    other_df = df[remaining_others]
    other_expanded, other_groups, _ = expand_terms(other_df, nk=nk)
    other_cols = [c for g in other_groups.values() for c in g]
    X = other_expanded[other_cols].values

    results = {}
    for dcol in dummies.columns:
        y = dummies[dcol].values
        results[dcol] = ols_r2(y, X, adjusted=adjusted)

    return results, all(v >= r2_threshold for v in results.values())