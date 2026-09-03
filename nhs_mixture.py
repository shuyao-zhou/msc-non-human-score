# =====================================================================
# nhs_mixture.py — Custom mixture models for the Non-Human Score
# ---------------------------------------------------------------------
# Component structure (features are conditionally independent given the
# latent component):
#
#   volume          ~ left-truncated Negative Binomial (v >= V_MIN; default 20)
#   fanout          ~ zero-truncated Negative Binomial (f >= 1)
#   iat_cv          ~ Gamma
#   period_strength ~ two modes controlled by `period_null`:
#       "estimated" (default): both H and M use Gamma distributions with
#                   left censoring at S_MIN and right censoring at S_MAX=300.
#                   Components are oriented by fitted mean periodicity.
#       "anchored": H uses a fixed Exp(lambda0) calibration null. This was
#                   available as an alternative two-component specification.
#   missing iat_cv / period_strength -> omit that likelihood contribution
#   quiet-event count k ~ Beta-Binomial(v, mu_q*phi_q, (1-mu_q)*phi_q),
#                   where k = round(quiet_frac * volume).
#
# The final dissertation model uses the K-component implementation in
# `fit_mixture_k` / `score_mixture_k`.
#
# Dependencies: numpy, scipy, pandas; scikit-learn is used only for K-means
# initialisation.
# =====================================================================
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy import optimize, special
from scipy.special import gammaln, digamma, logsumexp
from scipy.stats import nbinom, gamma as gamma_dist

# ---------------- Global constants shared with the main pipeline ----------------
# Stable identifier for the K-component implementation used by the final pipeline.
MIXTURE_MODEL_VERSION = "custom_mixture_k_v1"

V_MIN   = 20          # minimum retained volume (MIN_AGG)
F_MIN   = 1           # zero-truncation point for fanout
S_MAX   = 300.0       # right-censoring cap for period_strength
S_MIN   = 0.05        # left-censoring threshold for period_strength
LAMBDA0 = np.log(10)  # rate of the anchored H-component Exp(ln10) null
EPS     = 1e-12


# =====================================================================
# 1. Feature-wise log-likelihood terms
# =====================================================================

def _nb_logpmf(x, mu, r):
    """Negative-Binomial log PMF in mean-dispersion parameterisation."""
    p = r / (r + mu)                      # scipy uses the (n, p) convention
    return nbinom.logpmf(x, r, p)


def _nb_logsf_below(vmin, mu, r):
    """Log truncation normaliser: log P(X >= vmin)."""
    p = r / (r + mu)
    # scipy sf(k) = P(X > k), so P(X >= vmin) = sf(vmin - 1)
    return np.log(np.maximum(nbinom.sf(vmin - 1, r, p), EPS))


def loglik_volume(v, mu, r, vmin=V_MIN):
    """Left-truncated Negative-Binomial log likelihood."""
    return _nb_logpmf(v, mu, r) - _nb_logsf_below(vmin, mu, r)


def loglik_fanout(f, mu, r):
    """Zero-truncated Negative-Binomial log likelihood for f >= 1."""
    return _nb_logpmf(f, mu, r) - _nb_logsf_below(F_MIN, mu, r)


def loglik_cv(c, alpha, beta):
    """Gamma log density using a rate parameterisation."""
    return (alpha * np.log(beta) - gammaln(alpha)
            + (alpha - 1.0) * np.log(np.maximum(c, EPS)) - beta * c)


def loglik_period_exp(s, obs, cen, lo, lam0=LAMBDA0):
    """Anchored H component: fixed Exp(lam0) with left/right censoring."""
    out = np.zeros_like(s, dtype=float)
    out[obs] = np.log(lam0) - lam0 * s[obs]
    out[cen] = -lam0 * S_MAX                          # log P(S>=300)
    out[lo]  = np.log(max(1.0 - np.exp(-lam0 * S_MIN), EPS))  # log P(S<=S_MIN)
    return out


def loglik_period_gamma(s, a, b, obs, cen, lo):
    """Gamma(a,b) likelihood with left censoring at S_MIN and right censoring at S_MAX."""
    out = np.zeros_like(s, dtype=float)
    out[obs] = loglik_cv(s[obs], a, b)
    out[cen] = np.log(np.maximum(gamma_dist.sf(S_MAX, a, scale=1.0 / b), EPS))
    out[lo]  = np.log(np.maximum(gamma_dist.cdf(S_MIN, a, scale=1.0 / b), EPS))
    return out


def loglik_quiet(k, v, mu, phi):
    """Beta-Binomial log likelihood, omitting the parameter-free combinatorial term.

    k ~ BetaBin(v, a=mu*phi, b=(1-mu)*phi).
    As phi -> infinity the distribution approaches Binomial; smaller phi
    permits stronger overdispersion.
    """
    mu = np.clip(mu, 1e-6, 1 - 1e-6)
    a, b = mu * phi, (1.0 - mu) * phi
    return special.betaln(k + a, v - k + b) - special.betaln(a, b)


def _binom_const(k, v):
    """Binomial combinatorial term log C(v,k), added only to reported total log likelihood."""
    return gammaln(v + 1) - gammaln(k + 1) - gammaln(v - k + 1)


# =====================================================================
# 2. Data preparation
# =====================================================================

def prepare(feat: pd.DataFrame) -> dict:
    """Extract arrays required by the mixture model.

    Required columns: volume, fanout, iat_cv, quiet_frac, period_strength.
    Raw volume/fanout counts are used rather than their log-transformed CORE5
    counterparts.
    """
    v = feat["volume"].to_numpy(dtype=float)
    f = feat["fanout"].to_numpy(dtype=float)
    c = feat["iat_cv"].to_numpy(dtype=float)
    s = feat["period_strength"].to_numpy(dtype=float)
    q = feat["quiet_frac"].to_numpy(dtype=float)

    # Missing iat_cv values are handled by omitting that likelihood term.
    # A 1e-3 floor avoids the Gamma singularity for exactly regular spacing.
    cv_obs = np.isfinite(c)
    c_safe = np.maximum(np.where(cv_obs, c, 1.0), 1e-3)

    # Four-state periodicity mask: missing / right-censored / left-censored / observed.
    mis = ~np.isfinite(s)
    cen = np.isfinite(s) & (s >= S_MAX - 1e-9)
    lo  = np.isfinite(s) & ~cen & (s < S_MIN)
    obs = np.isfinite(s) & ~cen & ~lo
    s_safe = np.clip(np.where(np.isfinite(s), s, 0.0), 0.0, S_MAX)

    # Quiet-event count.
    k = np.rint(np.clip(q, 0, 1) * v)

    return dict(v=v, f=f, c=c_safe, cv_obs=cv_obs,
                s=s_safe, s_obs=obs, s_cen=cen, s_lo=lo, s_mis=mis,
                k=k, n=len(feat))


# =====================================================================
# 3. Component log likelihood (sum of five feature terms)
# =====================================================================

def component_loglik(d: dict, p: dict, which: str) -> np.ndarray:
    """Return log p_j(x_a) for component j in {H, M}."""
    j = p[which]
    ll = loglik_volume(d["v"], j["mu_v"], j["r_v"])
    ll += loglik_fanout(d["f"], j["mu_f"], j["r_f"])
    ll += np.where(d["cv_obs"], loglik_cv(d["c"], j["a_c"], j["b_c"]), 0.0)
    if which == "H" and p.get("period_null", "estimated") == "anchored":
        ll += loglik_period_exp(d["s"], d["s_obs"], d["s_cen"], d["s_lo"],
                                lam0=p.get("lambda0", LAMBDA0))
    else:
        ll += loglik_period_gamma(d["s"], j["a_s"], j["b_s"],
                                  d["s_obs"], d["s_cen"], d["s_lo"])
    ll += loglik_quiet(d["k"], d["v"], j["mu_q"], j["phi_q"])
    return ll


# =====================================================================
# 4. Weighted M-step estimators
# =====================================================================

def _fit_trunc_nb(x, w, vmin, init):
    """Weighted MLE for a truncated Negative Binomial; optimise (log mu, log r)."""
    w = np.asarray(w, dtype=float)
    W = w.sum()
    if W < EPS:
        return init

    mu_lo, mu_hi = max(0.2 * vmin, 0.5), 50.0 * float(np.max(x))
    r_lo, r_hi = 1e-3, 1e5

    def neg(theta):
        mu = np.clip(np.exp(theta[0]), mu_lo, mu_hi)
        r = np.clip(np.exp(theta[1]), r_lo, r_hi)
        ll = _nb_logpmf(x, mu, r) - _nb_logsf_below(vmin, mu, r)
        return -(w * ll).sum() / W

    x0 = np.log([np.clip(init[0], mu_lo, mu_hi), np.clip(init[1], r_lo, r_hi)])
    res = optimize.minimize(neg, x0, method="Nelder-Mead",
                            options=dict(xatol=1e-4, fatol=1e-7, maxiter=400))
    mu = float(np.clip(np.exp(res.x[0]), mu_lo, mu_hi))
    r = float(np.clip(np.exp(res.x[1]), r_lo, r_hi))
    return mu, r


def _fit_gamma_weighted(x, w, init):
    """Weighted Gamma MLE without censoring."""
    w = np.asarray(w, dtype=float)
    W = w.sum()
    if W < EPS:
        return init
    m = (w * x).sum() / W
    l = (w * np.log(np.maximum(x, EPS))).sum() / W
    gap = max(np.log(max(m, EPS)) - l, 1e-8)

    def eq(a):
        return np.log(a) - digamma(a) - gap

    try:
        a = optimize.brentq(eq, 1e-3, 1e6)
    except ValueError:
        a = init[0]
    b = a / max(m, EPS)
    return float(a), float(b)


def _fit_gamma_censored(s, w, obs, cen, lo, init):
    """Weighted censored-Gamma MLE for periodicity."""
    Wo, Wc, Wl = w[obs].sum(), w[cen].sum(), w[lo].sum()
    W = Wo + Wc + Wl
    if W < EPS:
        return init
    x_obs, w_obs = s[obs], w[obs]

    def neg(theta):
        a = np.clip(np.exp(theta[0]), 1e-3, 1e3)
        b = np.clip(np.exp(theta[1]), 1e-6, 1e3)
        ll = (w_obs * loglik_cv(x_obs, a, b)).sum()
        if Wc > 0:
            ll += Wc * np.log(np.maximum(
                gamma_dist.sf(S_MAX, a, scale=1.0 / b), EPS))
        if Wl > 0:
            ll += Wl * np.log(np.maximum(
                gamma_dist.cdf(S_MIN, a, scale=1.0 / b), EPS))
        return -ll / W

    x0 = np.log(np.maximum(init, 1e-3))
    res = optimize.minimize(neg, x0, method="Nelder-Mead",
                            options=dict(xatol=1e-4, fatol=1e-7, maxiter=500))
    a = float(np.clip(np.exp(res.x[0]), 1e-3, 1e3))
    b = float(np.clip(np.exp(res.x[1]), 1e-6, 1e3))
    return a, b


def _fit_betabin(k, v, w, init):
    """Weighted Beta-Binomial MLE; optimise (logit mu, log phi)."""
    w = np.asarray(w, dtype=float)
    W = w.sum()
    if W < EPS:
        return init

    def neg(theta):
        mu = 1.0 / (1.0 + np.exp(-theta[0]))
        phi = np.exp(theta[1])
        mu = np.clip(mu, 1e-6, 1 - 1e-6)
        a, b = mu * phi, (1.0 - mu) * phi
        ll = special.betaln(k + a, v - k + b) - special.betaln(a, b)
        return -(w * ll).sum() / W

    mu0 = np.clip(init[0], 1e-4, 1 - 1e-4)
    x0 = np.array([np.log(mu0 / (1 - mu0)), np.log(max(init[1], 0.1))])
    res = optimize.minimize(neg, x0, method="Nelder-Mead",
                            options=dict(xatol=1e-4, fatol=1e-7, maxiter=500))
    mu = float(1.0 / (1.0 + np.exp(-res.x[0])))
    phi = float(np.clip(np.exp(res.x[1]), 1e-3, 1e7))
    return mu, phi


# =====================================================================
# 5. Initialisation
# =====================================================================

def _moment_init(d, mask, tag):
    """Construct moment-based initial component parameters for the selected subset."""
    v, f, c, k = d["v"][mask], d["f"][mask], d["c"][mask], d["k"][mask]
    cvo = d["cv_obs"][mask]

    def nb_mom(x, floor_r=0.05):
        m, var = x.mean(), x.var() + EPS
        r = m * m / max(var - m, m * 0.1)      # When var <= mean, fall back to a large dispersion parameter.
        return float(max(m, 1.0)), float(np.clip(r, floor_r, 1e4))

    mu_v, r_v = nb_mom(v)
    mu_f, r_f = nb_mom(f)
    cm = c[cvo]
    a_c, b_c = _fit_gamma_weighted(cm, np.ones_like(cm), (1.0, 1.0)) \
        if cvo.sum() > 5 else (2.0, 2.0)
    mu_q = float(np.clip(k.sum() / max(v.sum(), 1.0), 1e-4, 1 - 1e-4))
    # Initialise phi from the across-row variance of quiet fractions, with floor 2.
    frac = k / np.maximum(v, 1.0)
    var_f = max(frac.var(), 1e-6)
    phi_q = float(np.clip(mu_q * (1 - mu_q) / var_f - 1.0, 2.0, 1e5))
    out = dict(mu_v=mu_v, r_v=r_v, mu_f=mu_f, r_f=r_f,
               a_c=a_c, b_c=b_c, mu_q=mu_q, phi_q=phi_q)
    so = d["s_obs"] & mask
    sm = d["s"][so]
    if sm.size > 20:
        a_s, b_s = _fit_gamma_weighted(np.maximum(sm, S_MIN),
                                       np.ones_like(sm), (1.0, 0.05))
    else:
        a_s, b_s = (1.0, 0.02) if tag == "M" else (1.0, 0.2)
    out.update(a_s=a_s, b_s=b_s)
    return out


def init_params(d, feat=None, kmeans_labels=None, lambda0=LAMBDA0,
                period_null="estimated"):
    """Initialise the two-component model.

    Prefer supplied K-means labels; otherwise split approximately by
    periodicity strength.
    """
    if kmeans_labels is not None:
        lab = np.asarray(kmeans_labels)
        # Orient the machine-like cluster as the one with larger mean period_strength.
        s = np.where(np.isfinite(d["s"]), d["s"], 0.0)
        m0 = s[lab == 0].mean() if (lab == 0).any() else -np.inf
        m1 = s[lab == 1].mean() if (lab == 1).any() else -np.inf
        machine_mask = lab == (0 if m0 > m1 else 1)
    else:
        s = np.where(d["s_mis"], -1.0, d["s"])
        thr = np.nanmedian(s[s >= 0]) if (s >= 0).any() else 5.0
        machine_mask = s > max(thr, 5.0)
        if machine_mask.sum() < 10 or (~machine_mask).sum() < 10:
            machine_mask = s > np.nanquantile(s, 0.7)

    p = dict(pi=float(np.clip(machine_mask.mean(), 0.05, 0.95)),
             lambda0=lambda0, period_null=period_null,
             H=_moment_init(d, ~machine_mask, "H"),
             M=_moment_init(d, machine_mask, "M"))
    return p


# =====================================================================
# 6. Two-component EM algorithm
# =====================================================================

def fit_mixture(feat: pd.DataFrame,
                kmeans_labels=None,
                lambda0: float = LAMBDA0,
                period_null: str = "estimated",
                max_iter: int = 100,
                tol: float = 1e-6,
                verbose: bool = True) -> dict:
    """Fit the two-component custom mixture.

    Parameters
    ----------
    feat
        Feature table containing volume, fanout, iat_cv, quiet_frac and
        period_strength.
    kmeans_labels
        Optional K-means labels used only for initialisation.
    lambda0
        Rate of the anchored periodicity null when `period_null="anchored"`.

    Returns
    -------
    dict
        params, gamma, llr, loglik_trace, n_iter and converged.
    """
    d = prepare(feat)
    p = init_params(d, feat, kmeans_labels, lambda0, period_null)
    const = _binom_const(d["k"], d["v"]).sum()   # reporting constant

    trace = []
    prev = -np.inf
    for it in range(max_iter):
        # ---------- E step ----------
        llH = component_loglik(d, p, "H")
        llM = component_loglik(d, p, "M")
        a = np.log(p["pi"]) + llM
        b = np.log1p(-p["pi"]) + llH
        norm = np.logaddexp(a, b)
        gam = np.exp(a - norm)                    # responsibility / posterior mass
        ll = norm.sum() + const
        trace.append(ll)

        if verbose:
            print(f"iter {it:3d}  loglik = {ll:.2f}  pi = {p['pi']:.4f}")
        if ll < prev - 1e-6 * abs(prev):
            # Guard against a substantial violation of EM monotonicity.
            raise RuntimeError(
                f"log-likelihood decreased at iter {it}: {prev:.4f} -> {ll:.4f}")
        if it > 0 and abs(ll - prev) < tol * abs(prev):
            prev = ll
            break
        prev = ll

        # ---------- M step ----------
        wM, wH = gam, 1.0 - gam
        p["pi"] = float(np.clip(gam.mean(), 1e-4, 1 - 1e-4))
        for tag, w in (("H", wH), ("M", wM)):
            if w.sum() < max(20.0, 1e-4 * d["n"]):
                continue          # Freeze a starved component to avoid degeneracy.
            j = p[tag]
            j["mu_v"], j["r_v"] = _fit_trunc_nb(
                d["v"], w, V_MIN, (j["mu_v"], j["r_v"]))
            j["mu_f"], j["r_f"] = _fit_trunc_nb(
                d["f"], w, F_MIN, (j["mu_f"], j["r_f"]))
            cvw = w * d["cv_obs"]
            j["a_c"], j["b_c"] = _fit_gamma_weighted(
                d["c"], cvw, (j["a_c"], j["b_c"]))
            j["mu_q"], j["phi_q"] = _fit_betabin(
                d["k"], d["v"], w, (j["mu_q"], j["phi_q"]))
        # Update M periodicity always; update H only in estimated mode.
        p["M"]["a_s"], p["M"]["b_s"] = _fit_gamma_censored(
            d["s"], wM, d["s_obs"], d["s_cen"], d["s_lo"],
            (p["M"]["a_s"], p["M"]["b_s"]))
        if period_null == "estimated":
            p["H"]["a_s"], p["H"]["b_s"] = _fit_gamma_censored(
                d["s"], wH, d["s_obs"], d["s_cen"], d["s_lo"],
                (p["H"]["a_s"], p["H"]["b_s"]))

    # Canonical orientation: M is the component with larger fitted mean periodicity.
    if period_null == "estimated":
        mean_M = p["M"]["a_s"] / p["M"]["b_s"]
        mean_H = p["H"]["a_s"] / p["H"]["b_s"]
        if mean_H > mean_M:
            p["H"], p["M"] = p["M"], p["H"]
            p["pi"] = 1.0 - p["pi"]
            gam = 1.0 - gam
            llH, llM = llM, llH
            if verbose:
                print("[relabel] swapped components so M has the higher fitted periodicity mean")

    llr = llM - llH
    return dict(params=p, gamma=gam, llr=llr,
                loglik_trace=np.array(trace),
                n_iter=len(trace), converged=(len(trace) < max_iter))


# =====================================================================
# 7. Frozen-parameter scoring interfaces
# =====================================================================

def diagnose_contributions(feat: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Summarise absolute feature-wise LLR contributions for diagnostics."""
    d = prepare(feat)
    H, M = params["H"], params["M"]
    parts = {
        "volume": (loglik_volume(d["v"], M["mu_v"], M["r_v"])
                   - loglik_volume(d["v"], H["mu_v"], H["r_v"])),
        "fanout": (loglik_fanout(d["f"], M["mu_f"], M["r_f"])
                   - loglik_fanout(d["f"], H["mu_f"], H["r_f"])),
        "iat_cv": np.where(d["cv_obs"],
                           loglik_cv(d["c"], M["a_c"], M["b_c"])
                           - loglik_cv(d["c"], H["a_c"], H["b_c"]), 0.0),
        "period": (loglik_period_gamma(d["s"], M["a_s"], M["b_s"],
                                       d["s_obs"], d["s_cen"], d["s_lo"])
                   - (loglik_period_exp(d["s"], d["s_obs"], d["s_cen"],
                                        d["s_lo"],
                                        lam0=params.get("lambda0", LAMBDA0))
                      if params.get("period_null", "estimated") == "anchored"
                      else loglik_period_gamma(d["s"], H["a_s"], H["b_s"],
                                               d["s_obs"], d["s_cen"],
                                               d["s_lo"]))),
        "quiet": (loglik_quiet(d["k"], d["v"], M["mu_q"], M["phi_q"])
                  - loglik_quiet(d["k"], d["v"], H["mu_q"], H["phi_q"])),
    }
    rows = {}
    for name, llr in parts.items():
        a = np.abs(llr)
        rows[name] = dict(median=np.median(a),
                          q90=np.quantile(a, 0.90),
                          q99=np.quantile(a, 0.99),
                          max=a.max())
    return pd.DataFrame(rows).T.round(2)


def score_mixture(feat: pd.DataFrame, params: dict,
                  return_llr: bool = False) -> np.ndarray:
    """Score new rows with frozen two-component parameters.

    Returns posterior mass by default; with `return_llr=True`, returns the
    component log-likelihood ratio.
    """
    d = prepare(feat)
    llH = component_loglik(d, params, "H")
    llM = component_loglik(d, params, "M")
    if return_llr:
        return llM - llH
    a = np.log(params["pi"]) + llM
    b = np.log1p(-params["pi"]) + llH
    return np.exp(a - np.logaddexp(a, b))


def params_to_json(params: dict, path: str):
    with open(path, "w") as fh:
        json.dump(params, fh, indent=2)


def params_from_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


# =====================================================================
# 8. Legacy null-distribution bootstrap utility
# =====================================================================

def bootstrap_null(fisher_fn, horizon, rates=(0.001, 0.01, 0.1),
                   n_sim=300, seed=42):
    """Simulate non-periodic Poisson event streams through a supplied Fisher statistic.

    This utility provides an empirical reference distribution for the
    two-component periodicity specification.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for rate in rates:
        n_exp = rate * horizon
        for _ in range(n_sim):
            n = rng.poisson(n_exp)
            if n < 50:
                continue
            times = np.sort(rng.uniform(0, horizon, n)).astype(np.int64)
            stat = fisher_fn(times)
            if np.isfinite(stat):
                rows.append((rate, stat))
    return pd.DataFrame(rows, columns=["rate", "stat"])


# =====================================================================
# 9. Self-test utilities
# =====================================================================

def _make_synthetic(n_h=3000, n_m=800, seed=0):
    """Generate synthetic feature rows and labels for a basic implementation self-test."""
    rng = np.random.default_rng(seed)

    def draw_trunc_nb(n, mu, r, vmin):
        out = np.empty(n)
        filled = 0
        while filled < n:
            x = rng.negative_binomial(r, r / (r + mu), size=2 * n)
            x = x[x >= vmin]
            take = min(len(x), n - filled)
            out[filled:filled + take] = x[:take]
            filled += take
        return out

    # Human-like synthetic component: lower volume/fanout, larger CV, lower quiet activity.
    vH = draw_trunc_nb(n_h, 60, 0.8, V_MIN)
    fH = draw_trunc_nb(n_h, 3, 1.0, F_MIN)
    cH = rng.gamma(4.0, 1.0 / 1.5, n_h)          # mean ~2.7
    sH = rng.exponential(1 / LAMBDA0, n_h)
    qH = rng.binomial(vH.astype(int), 0.10) / vH
    # Machine-like synthetic component: larger volume, smaller CV and stronger periodicity.
    vM = draw_trunc_nb(n_m, 5000, 0.5, V_MIN)
    fM = draw_trunc_nb(n_m, 8, 0.7, F_MIN)
    cM = rng.gamma(3.0, 1.0 / 8.0, n_m)          # mean ~0.37
    sM = np.minimum(rng.gamma(1.2, 90.0, n_m), S_MAX)
    qM = rng.binomial(vM.astype(int), 0.33) / vM

    feat = pd.DataFrame(dict(
        volume=np.concatenate([vH, vM]),
        fanout=np.concatenate([fH, fM]),
        iat_cv=np.concatenate([cH, cM]),
        period_strength=np.concatenate([sH, sM]),
        quiet_frac=np.concatenate([qH, qM]),
    ))
    y = np.concatenate([np.zeros(n_h), np.ones(n_m)])
    # Random periodicity missingness for the self-test.
    mis = rng.random(len(feat)) < 0.3
    feat.loc[mis, "period_strength"] = np.nan
    return feat, y


# =====================================================================
# 10. K-component extension used in the final dissertation
# ---------------------------------------------------------------------
# Automated behaviour is heterogeneous, so the final model fits K latent
# components with the same feature distributions as above. Periodicity uses
# a censored Gamma in every component. Components are then ordered by fitted
# periodicity mean and a fixed subset of components is designated as
# automation-oriented after fitting-set profile inspection. The NHS is the
# posterior mass assigned to that fixed subset.
# =====================================================================

def _component_loglik_k(d, comp):
    """Log likelihood for one K-component model component."""
    ll = loglik_volume(d["v"], comp["mu_v"], comp["r_v"])
    ll += loglik_fanout(d["f"], comp["mu_f"], comp["r_f"])
    ll += np.where(d["cv_obs"], loglik_cv(d["c"], comp["a_c"], comp["b_c"]), 0.0)
    ll += loglik_period_gamma(d["s"], comp["a_s"], comp["b_s"],
                              d["s_obs"], d["s_cen"], d["s_lo"])
    ll += loglik_quiet(d["k"], d["v"], comp["mu_q"], comp["phi_q"])
    return ll


def _kmeans_init_labels(d, K, seed=42):
    """K-way K-means initialisation in a transformed feature space."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    X = np.column_stack([
        np.log1p(d["v"]), np.log1p(d["f"]),
        np.log(np.maximum(d["c"], 1e-3)),
        np.where(d["s_mis"], 0.0, d["s"]) / S_MAX,
        d["k"] / np.maximum(d["v"], 1.0),
    ])
    Xs = StandardScaler().fit_transform(X)
    return KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(Xs)


N_PARAMS_PER_COMP = 10   # mu_v r_v mu_f r_f a_c b_c mu_q phi_q a_s b_s


def fit_mixture_k(feat: pd.DataFrame, K: int = 4,
                  init_labels=None, seed: int = 42,
                  max_iter: int = 150, tol: float = 1e-6,
                  verbose: bool = True) -> dict:
    """Fit the K-component custom mixture used in the final dissertation.

    Returns
    -------
    dict
        params
            Mixture weights, component parameter dictionaries and K.
        resp
            n x K posterior responsibility matrix.
        summary
            Fitting-set component profile table used for interpretation.
        bic
            Bayesian Information Criterion for K selection.

    Components are canonically ordered by decreasing fitted periodicity mean,
    so component 0 has the largest fitted periodicity mean.
    """
    d = prepare(feat)
    n = d["n"]
    if init_labels is None:
        init_labels = _kmeans_init_labels(d, K, seed)
    comps, pis = [], []
    for kk in range(K):
        mask = np.asarray(init_labels) == kk
        if mask.sum() < 20:
            mask = np.ones(n, dtype=bool)
        comps.append(_moment_init(d, mask, "M"))
        pis.append(max(mask.mean(), 1e-3))
    pi = np.array(pis); pi = pi / pi.sum()
    const = _binom_const(d["k"], d["v"]).sum()

    trace, prev = [], -np.inf
    for it in range(max_iter):
        LL = np.column_stack([_component_loglik_k(d, c) for c in comps])
        A = np.log(pi)[None, :] + LL
        norm = logsumexp(A, axis=1)
        R = np.exp(A - norm[:, None])
        ll = norm.sum() + const
        trace.append(ll)
        if verbose:
            print(f"iter {it:3d}  loglik = {ll:.2f}  pi = "
                  + np.array2string(pi, precision=3))
        if ll < prev - 1e-6 * abs(prev):
            raise RuntimeError(f"loglik decreased at iter {it}")
        if it > 0 and abs(ll - prev) < tol * abs(prev):
            prev = ll
            break
        prev = ll

        pi = np.clip(R.mean(axis=0), 1e-5, 1.0)
        pi = pi / pi.sum()
        for kk in range(K):
            w = R[:, kk]
            if w.sum() < max(20.0, 1e-4 * n):
                continue
            c = comps[kk]
            c["mu_v"], c["r_v"] = _fit_trunc_nb(d["v"], w, V_MIN,
                                                (c["mu_v"], c["r_v"]))
            c["mu_f"], c["r_f"] = _fit_trunc_nb(d["f"], w, F_MIN,
                                                (c["mu_f"], c["r_f"]))
            c["a_c"], c["b_c"] = _fit_gamma_weighted(
                d["c"], w * d["cv_obs"], (c["a_c"], c["b_c"]))
            c["mu_q"], c["phi_q"] = _fit_betabin(
                d["k"], d["v"], w, (c["mu_q"], c["phi_q"]))
            c["a_s"], c["b_s"] = _fit_gamma_censored(
                d["s"], w, d["s_obs"], d["s_cen"], d["s_lo"],
                (c["a_s"], c["b_s"]))

    # Canonical ordering by decreasing fitted periodicity mean.
    order = np.argsort([-c["a_s"] / c["b_s"] for c in comps])
    comps = [comps[i] for i in order]
    pi = pi[order]
    LL = np.column_stack([_component_loglik_k(d, c) for c in comps])
    A = np.log(pi)[None, :] + LL
    R = np.exp(A - logsumexp(A, axis=1)[:, None])

    n_par = K * N_PARAMS_PER_COMP + (K - 1)
    bic = -2.0 * prev + n_par * np.log(n)

    rows = []
    for kk, c in enumerate(comps):
        w = R[:, kk]; W = w.sum()
        rows.append(dict(
            comp=kk, pi=round(float(pi[kk]), 4),
            period_mean=round(c["a_s"] / c["b_s"], 1),
            pct_capped=round(float((w * d["s_cen"]).sum() / max(W, 1)), 3),
            volume_mean=round(c["mu_v"], 0),
            fanout_mean=round(c["mu_f"], 1),
            cv_mean=round(c["a_c"] / c["b_c"], 2),
            quiet_mu=round(c["mu_q"], 3),
            quiet_phi=round(c["phi_q"], 1),
        ))
    summary = pd.DataFrame(rows)

    return dict(params=dict(pi=pi.tolist(), comps=comps, K=K),
                resp=R, summary=summary, bic=float(bic),
                loglik_trace=np.array(trace), n_iter=len(trace),
                converged=(len(trace) < max_iter))


def merge_score(res_or_resp, machine_comps) -> np.ndarray:
    """Return the posterior mass assigned to the specified automation-oriented components."""
    R = res_or_resp["resp"] if isinstance(res_or_resp, dict) else res_or_resp
    return R[:, list(machine_comps)].sum(axis=1)


def score_mixture_k(feat: pd.DataFrame, params: dict, machine_comps,
                    return_llr: bool = False) -> np.ndarray:
    """Score new rows with frozen K-component parameters.

    By default returns posterior mass assigned to `machine_comps`.
    With `return_llr=True`, returns the marginal log-likelihood ratio between
    the automation-oriented and remaining component sets.
    """
    d = prepare(feat)
    pi = np.asarray(params["pi"])
    LL = np.column_stack([_component_loglik_k(d, c) for c in params["comps"]])
    A = np.log(pi)[None, :] + LL
    mach = np.zeros(params["K"], dtype=bool)
    mach[list(machine_comps)] = True
    lm = logsumexp(A[:, mach], axis=1)
    lh = logsumexp(A[:, ~mach], axis=1)
    if return_llr:
        return lm - lh
    return np.exp(lm - np.logaddexp(lm, lh))


def _selftest_realistic(seed=5):
    """Generate a more realistic two-component synthetic self-test scenario."""
    rng = np.random.default_rng(seed)

    def tnb(n, mu, r, vmin):
        out = []
        while len(out) < n:
            x = rng.negative_binomial(r, r / (r + mu), 3 * n)
            out += list(x[x >= vmin])
        return np.array(out[:n], float)

    n_h, n_m = 7000, 3000
    vH = tnb(n_h, 1500, 0.4, V_MIN); vM = tnb(n_m, 5000, 0.6, V_MIN)
    fH = tnb(n_h, 10, 3.0, 1);       fM = tnb(n_m, 11, 3.5, 1)
    cH = rng.gamma(2.5, 1 / 0.6, n_h); cM = rng.gamma(1.0, 1 / 0.14, n_m)
    sH = np.minimum(rng.gamma(1.8, 25.0, n_h), S_MAX)     # background periodicity: mean 45
    sM = np.minimum(rng.gamma(0.7, 320.0, n_m), S_MAX)    # stronger automation-like periodicity with substantial censoring
    sH += rng.normal(0, 0.5, n_h)                          # add small negative noise
    pH = rng.beta(1.5, 8, n_h); pM = rng.beta(2.5, 6, n_m)
    qH = rng.binomial(vH.astype(int), pH) / vH
    qM = rng.binomial(vM.astype(int), pM) / vM
    feat = pd.DataFrame(dict(volume=np.r_[vH, vM], fanout=np.r_[fH, fM],
                             iat_cv=np.r_[cH, cM],
                             period_strength=np.r_[sH, sM],
                             quiet_frac=np.r_[qH, qM]))
    y = np.r_[np.zeros(n_h), np.ones(n_m)]
    mis = rng.random(len(feat)) < 0.05
    feat.loc[mis, "period_strength"] = np.nan
    return feat, y


if __name__ == "__main__":
    print("== Test 1: clean separable synthetic data (anchored) ==")
    feat, y = _make_synthetic()
    res = fit_mixture(feat, period_null="anchored", verbose=True)
    gam = res["gamma"]
    from sklearn.metrics import roc_auc_score, average_precision_score
    print("\n=== Self-test results ===")
    print(f"converged: {res['converged']}  iters: {res['n_iter']}")
    print(f"pi_hat = {res['params']['pi']:.3f}  (true {y.mean():.3f})")
    print(f"AUROC  = {roc_auc_score(y, gam):.4f}")
    print(f"AUPRC  = {average_precision_score(y, gam):.4f}")
    print(f"mean gamma | true H = {gam[y == 0].mean():.3f}  "
          f"| true M = {gam[y == 1].mean():.3f}")
    print("\nMachine-component parameters:", {k: round(v, 3) for k, v in res['params']['M'].items()})
    print("Human-component parameters:", {k: round(v, 3) for k, v in res['params']['H'].items()})

    print("\n== Test 2: more realistic synthetic scenario (estimated) ==")
    feat2, y2 = _selftest_realistic()
    res2 = fit_mixture(feat2, period_null="estimated", verbose=False)
    g2 = res2["gamma"]
    print(f"converged: {res2['converged']}  iters: {res2['n_iter']}")
    print(f"pi_hat = {res2['params']['pi']:.3f}  (true {y2.mean():.3f})")
    print(f"AUROC  = {roc_auc_score(y2, g2):.4f}")
    pm2, ph2 = res2["params"]["M"], res2["params"]["H"]
    print(f"period means: M={pm2['a_s']/pm2['b_s']:.1f}  H={ph2['a_s']/ph2['b_s']:.1f}"
          f"  (approximately M~224 before censoring, H~45)")