import arviz as az
import numpy as np
import pandas as pd
from scipy import stats


def hoeffding_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    R = stats.rankdata(x)
    S = stats.rankdata(y)
    # bivariate rank: for each point, how many others lie below it on both axes
    Q = np.ones(n)
    for i in range(n):
        Q[i] += np.sum((x < x[i]) & (y < y[i]))
    D1 = np.sum((Q - 1) * (Q - 2))
    D2 = np.sum((R - 1) * (R - 2) * (S - 1) * (S - 2))
    D3 = np.sum((R - 2) * (S - 2) * (Q - 1))
    return 30 * ((n - 2) * (n - 3) * D1 + D2 - 2 * (n - 2) * D3) / (
        n * (n - 1) * (n - 2) * (n - 3) * (n - 4)
    )


def compute_t_stat(first_sample, second_sample, center=0):
    mean_diff = np.mean(first_sample) - np.mean(second_sample)
    se_diff = np.sqrt(np.var(first_sample, ddof=1) / len(first_sample) + np.var(second_sample, ddof=1) / len(second_sample))
    return (mean_diff - center) / se_diff


def neg_log_likelihood(params, data):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    return -np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

def bootstrap_mcfadden_r2(X, y, model_predict):
    """Compute McFadden pseudo R2 for predictions on given data."""
    p = model_predict(X)
    ll = np.sum(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
    ll_null_val = y.mean()
    ll_null = np.sum(y * np.log(ll_null_val) + (1 - y) * np.log(1 - ll_null_val))
    return 1 - ll / ll_null

def calibration_by(predictor, outcome, predicted_prob, n_bins=10):
    """Bin observations by a predictor and compare the observed event rate to the
    model's mean predicted probability within each bin.

    Parameters
    ----------
    predictor      : array-like - predictor values used to form the bins
    outcome        : array-like - binary outcomes (0/1)
    predicted_prob : array-like - model predicted probabilities for each observation
    n_bins         : int        - number of quantile bins (default: 10)

    Returns
    -------
    pandas.DataFrame with one row per bin and the columns:
        bin_center, bin_low, bin_high, observed_rate, predicted_rate, standard_error
    """
    predictor = np.asarray(predictor)
    outcome = np.asarray(outcome)
    predicted_prob = np.asarray(predicted_prob)

    bin_edges = np.quantile(predictor, np.linspace(0, 1, n_bins + 1))
    bin_index = np.clip(np.digitize(predictor, bin_edges[1:-1]), 0, n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        mask = bin_index == bin_id
        if not mask.any():
            continue
        count = mask.sum()
        rate = outcome[mask].mean()
        rows.append({
            "bin_center": predictor[mask].mean(),
            "bin_low": bin_edges[bin_id],
            "bin_high": bin_edges[bin_id + 1],
            "observed_rate": rate,
            "predicted_rate": predicted_prob[mask].mean(),
            "standard_error": np.sqrt(rate * (1 - rate) / count),
        })
    return pd.DataFrame(rows)

def normal_normal_posterior(
    observations,
    prior_mean,
    prior_variance,
    observation_variance,
    hdi_prob=0.95,
    n_samples=100_000,
    random_state=42
):
    """
    Compute the analytical posterior for a Normal-Normal conjugate model.

    Parameters
    ----------
    observations         : array-like - observed data points
    prior_mean           : float      - mean of the Normal prior
    prior_variance       : float      - variance of the Normal prior
    observation_variance : float      - known variance of the likelihood
    hdi_prob             : float      - probability mass for the HDI (default: 0.95)
    n_samples            : int        - number of samples drawn to estimate the HDI
    random_state         : int        - random seed for reproducibility

    Returns
    -------
    dict with posterior mean, variance, std, and HDI bounds
    """
    x = np.asarray(observations)
    n = len(x)
    x_bar = x.mean()

    post_var = 1 / (1 / prior_variance + n / observation_variance)
    post_mean = post_var * (
        prior_mean / prior_variance +
        n * x_bar / observation_variance
    )
    post_std = np.sqrt(post_var)

    samples = stats.norm.rvs(loc=post_mean, scale=post_std,
                             size=n_samples, random_state=random_state)
    hdi = az.hdi(samples, prob=hdi_prob)

    print(f"Posterior        : N({post_mean:.4f}, {post_std:.4f}²)")
    print(f"Posterior mean   : {post_mean:.4f}")
    print(f"{int(hdi_prob*100)}% HDI          : ({hdi[0]:.4f}, {hdi[1]:.4f})")

    return {"mean": post_mean, "variance": post_var, "std": post_std,
            "hdi_low": hdi[0], "hdi_high": hdi[1]}