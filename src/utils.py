import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(df):
    vif_df = pd.DataFrame()
    vif_df["feature"] = df.columns
    vif_df["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_df

def likelihood_ratio_test(full_model, reduced_model):
    LL_full = full_model.llf
    LL_reduced = reduced_model.llf

    df_diff = full_model.df_model - reduced_model.df_model
    LR_stat = 2 * (LL_full - LL_reduced)
    p_value = stats.chi2.sf(LR_stat, df_diff)

    print(f"LR stat: {LR_stat:.3f}, df diff: {df_diff}, p-value: {p_value:.4f}")
    return LR_stat, p_value


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


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


def check_linearity(x, y, df, q = 4):
    df_temp = df.copy()
    
    df_temp["x_bin"] = pd.qcut(df_temp[x], q=q, duplicates='drop')

    summary = df_temp.groupby("x_bin").agg(
        median_x = (x, "median"),
        p_hat = (y, "mean")
    ).reset_index()

    summary["p_hat"] = summary["p_hat"].clip(0.01, 0.99)
    summary["log_odds"] = np.log(summary["p_hat"] / (1 - summary["p_hat"]))

    plt.figure()
    plt.scatter(summary["median_x"], summary["log_odds"], color='steelblue')
    plt.plot(summary["median_x"], summary["log_odds"], linestyle='--', color='gray')
    plt.xlabel(f"Median of {x}")
    plt.ylabel("Logit(p)")
    plt.title(f"{x} vs Log odds of {y}")
    plt.grid(True)
    plt.show()

    return summary

def compute_t_stat(first_sample, second_sample):
    mean_diff = np.mean(first_sample) - np.mean(second_sample)
    se_diff = np.sqrt(np.var(first_sample, ddof=1) / len(first_sample) + np.var(second_sample, ddof=1) / len(second_sample))
    return mean_diff / se_diff


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

def beta_binom_posterior(k, n, alpha_prior=1, beta_prior=1, hdi_prob=0.95, n_samples=100_000, random_state=42):
    """
    Compute the analytical posterior for a Beta-Binomial conjugate model.

    Parameters
    ----------
    k            : int   - number of successes observed
    n            : int   - number of trials
    alpha_prior  : float - alpha parameter of the Beta prior (default: 1)
    beta_prior   : float - beta parameter of the Beta prior (default: 1)
    hdi_prob     : float - probability mass for the HDI (default: 0.95)
    n_samples    : int   - number of samples drawn to estimate the HDI
    random_state : int   - random seed for reproducibility

    Returns
    -------
    dict with posterior alpha/beta, mean, and HDI bounds
    """
    alpha_post = alpha_prior + k
    beta_post  = beta_prior  + (n - k)

    posterior  = stats.beta(alpha_post, beta_post)
    post_mean  = posterior.mean()

    samples = posterior.rvs(size=n_samples, random_state=random_state)
    hdi     = az.hdi(samples, prob=hdi_prob)

    print(f"Posterior        : Beta({alpha_post}, {beta_post})")
    print(f"Posterior mean   : {post_mean:.4f}")
    print(f"{int(hdi_prob*100)}% HDI          : ({hdi[0]:.4f}, {hdi[1]:.4f})")

    return {"alpha_post": alpha_post, "beta_post": beta_post,
            "mean": post_mean, "hdi_low": hdi[0], "hdi_high": hdi[1]}


def compute_posterior(n, k, p0=0.1, p1=0.2, prior_h0=0.5, prior_h1=0.5):
    """
    Discrete Bayesian hypothesis test between two point hypotheses.

    Parameters
    ----------
    n        : int   - number of trials
    k        : int   - number of successes observed
    p0       : float - success probability under H0 (default: 0.1)
    p1       : float - success probability under H1 (default: 0.2)
    prior_h0 : float - prior probability of H0 (default: 0.5)
    prior_h1 : float - prior probability of H1 (default: 0.5)

    Returns
    -------
    (posterior_h0, posterior_h1) : tuple of floats
    """
    likelihood_h0 = stats.binom.pmf(k, n, p0)
    likelihood_h1 = stats.binom.pmf(k, n, p1)

    evidence = (prior_h0 * likelihood_h0) + (prior_h1 * likelihood_h1)

    posterior_h0 = prior_h0 * likelihood_h0 / evidence
    posterior_h1 = prior_h1 * likelihood_h1 / evidence

    return posterior_h0, posterior_h1


def gamma_poisson_posterior(
    dataset,
    prior_mu,
    prior_sigma,
    hdi_prob=0.95,
    n_samples=100_000,
    random_state=42
):
    """
    Compute the analytical posterior for a Gamma-Poisson conjugate model.

    Uses mean/sigma parameterization for the Gamma prior (matching PyMC convention):
        shape (α) = prior_mu² / prior_sigma²
        rate  (β) = prior_mu  / prior_sigma²

    Conjugate update:
        Posterior: Gamma(α + sum(x), rate = β + n)

    Parameters
    ----------
    dataset      : array-like - observed count data
    prior_mu     : float      - prior mean (μ) of the Gamma distribution
    prior_sigma  : float      - prior std  (σ) of the Gamma distribution
    hdi_prob     : float      - probability mass for the HDI (default: 0.95)
    n_samples    : int        - number of samples drawn to estimate the HDI
    random_state : int        - random seed for reproducibility

    Returns
    -------
    dict with posterior shape, rate, mean, std, and HDI bounds
    """
    data = np.asarray(dataset)
    n = len(data)

    # Convert mu/sigma → shape/rate
    prior_shape = prior_mu**2 / prior_sigma**2
    prior_rate  = prior_mu  / prior_sigma**2

    # Conjugate update
    post_shape = prior_shape + data.sum()
    post_rate  = prior_rate  + n
    post_mean  = post_shape / post_rate
    post_std   = np.sqrt(post_shape) / post_rate

    # HDI via sampling from posterior Gamma (shape, scale=1/rate)
    samples = stats.gamma.rvs(post_shape, scale=1 / post_rate,
                               size=n_samples, random_state=random_state)
    hdi = az.hdi(samples, prob=hdi_prob)

    print(f"Posterior        : Gamma(shape={post_shape:.4f}, rate={post_rate:.4f})")
    print(f"Posterior mean   : {post_mean:.4f}")
    print(f"Posterior std    : {post_std:.4f}")
    print(f"{int(hdi_prob*100)}% HDI          : ({hdi[0]:.4f}, {hdi[1]:.4f})")

    return {"shape": post_shape, "rate": post_rate,
            "mean": post_mean, "std": post_std,
            "hdi_low": hdi[0], "hdi_high": hdi[1]}