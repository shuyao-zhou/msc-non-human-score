"""
Reusable utilities for the static Non-Human Score pipeline.

This module contains account splitting, weak-proxy labelling, ranking metrics,
and the three generic frozen scorers used in the final dissertation:
empirical rank, K-means and a two-component Gaussian mixture model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# Among the retained CORE5 features, only IAT-CV is oriented in reverse:
# lower raw IAT-CV means more regular, machine-like spacing.
REVERSE_FEATURES = {"iat_cv"}


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weak proxy labels used only for evaluation and interpretation.

    kind:
        "machine" for dollar-suffix computer-account proxies,
        "human" for U-named user-account proxies,
        "other" otherwise.

    is_anchor:
        broader automation-associated reference consisting of dollar-suffix
        accounts plus SYSTEM, LOCAL SERVICE and NETWORK SERVICE.
    """
    out = df.copy()
    u = out["u"].astype(str)

    out["kind"] = np.where(
        u.str.contains(r"\$@", regex=True),
        "machine",
        np.where(u.str.startswith("U"), "human", "other"),
    )

    out["is_anchor"] = (
        u.str.contains(r"\$@", regex=True)
        | u.str.contains(
            r"SYSTEM|LOCAL SERVICE|NETWORK SERVICE",
            case=False,
            regex=True,
        )
    )
    return out


def split_accounts(
    df: pd.DataFrame,
    fracs=(0.60, 0.15, 0.25),
    seed: int = 42,
):
    """Split unique accounts into fit, calibration and test sets."""
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError("fracs must sum to 1")

    rng = np.random.default_rng(seed)
    accounts = np.array(sorted(df["u"].astype(str).unique()))
    rng.shuffle(accounts)

    n = len(accounts)
    n_fit = int(fracs[0] * n)
    n_cal = int(fracs[1] * n)

    fit = set(accounts[:n_fit])
    cal = set(accounts[n_fit:n_fit + n_cal])
    test = set(accounts[n_fit + n_cal:])
    return fit, cal, test


def save_account_split(
    path: str | Path,
    fit_a: set,
    cal_a: set,
    te_a: set,
) -> None:
    """Write the fixed account-level split as JSON."""
    path = Path(path)
    obj = {
        "fit": sorted(map(str, fit_a)),
        "calibration": sorted(map(str, cal_a)),
        "test": sorted(map(str, te_a)),
    }
    path.write_text(json.dumps(obj, indent=2))


def precision_at_k(scores, is_pos, k: int) -> float:
    """Precision among the k largest scores."""
    scores = np.asarray(scores)
    is_pos = np.asarray(is_pos, dtype=bool)

    if len(scores) == 0:
        return np.nan

    k = min(int(k), len(scores))
    order = np.argsort(-scores)[:k]
    return float(is_pos[order].mean())


def recall_at_k(scores, is_pos, k: int) -> float:
    """Fraction of all positive references retrieved among the k largest scores."""
    scores = np.asarray(scores)
    is_pos = np.asarray(is_pos, dtype=bool)

    if len(scores) == 0 or is_pos.sum() == 0:
        return np.nan

    k = min(int(k), len(scores))
    order = np.argsort(-scores)[:k]
    return float(is_pos[order].sum() / is_pos.sum())


def _prepare_matrix(
    df: pd.DataFrame,
    cols: Sequence[str],
    medians=None,
    scaler=None,
):
    """Orient CORE5, median-impute and standardise using frozen fit quantities."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing model features: {missing}")

    X = df[list(cols)].copy()

    for c in cols:
        if c in REVERSE_FEATURES:
            X[c] = -X[c]

    if medians is None:
        medians = X.median(numeric_only=True)

    X = X.fillna(medians)

    if scaler is None:
        scaler = StandardScaler().fit(X)

    Xs = scaler.transform(X)
    return Xs, X, medians, scaler


def _machine_cluster_id(
    labels,
    fit_df: pd.DataFrame,
    k: int,
) -> int:
    """
    Orient an unsupervised fit without using proxy labels.

    The component/cluster with the largest fitting-set mean period_strength is
    treated as the machine-oriented side.
    """
    labels = np.asarray(labels)

    best = 0
    best_mean = -np.inf

    for c in range(k):
        mean_period = fit_df.loc[
            labels == c,
            "period_strength",
        ].mean()

        if pd.isna(mean_period):
            mean_period = -np.inf

        if mean_period > best_mean:
            best = c
            best_mean = mean_period

    return int(best)


class FrozenBaseline:
    """
    Equal-weight empirical-rank reference.

    Each oriented feature is mapped through its fitting-set empirical CDF and
    the five resulting ranks are averaged. Missing values use the fitting-set
    median on the oriented scale.
    """

    def __init__(self, cols: Sequence[str]):
        self.cols = list(cols)
        self.reference = None

    def fit(self, df: pd.DataFrame):
        self.reference = {}

        for c in self.cols:
            x = df[c].copy()

            if c in REVERSE_FEATURES:
                x = -x

            self.reference[c] = (
                x.dropna()
                .sort_values()
                .to_numpy()
            )

        return self

    def score(self, df: pd.DataFrame):
        if self.reference is None:
            raise RuntimeError("FrozenBaseline has not been fitted")

        parts = []

        for c in self.cols:
            ref = self.reference[c]
            x = df[c].copy()

            if c in REVERSE_FEATURES:
                x = -x

            if len(ref) == 0:
                pct = np.full(len(df), 0.5)
            else:
                fill = float(np.nanmedian(ref))
                vals = x.fillna(fill).to_numpy()
                pct = (
                    np.searchsorted(ref, vals, side="right")
                    / len(ref)
                )

            parts.append(pct)

        return np.nanmean(np.vstack(parts), axis=0)


class FrozenKMeans:
    """Two-cluster K-means scorer frozen after fitting-set estimation."""

    def __init__(
        self,
        cols: Sequence[str],
        n_clusters: int = 2,
        seed: int = 42,
    ):
        self.cols = list(cols)
        self.n_clusters = int(n_clusters)
        self.seed = int(seed)

    def fit(self, df: pd.DataFrame):
        Xs, _, self.medians_, self.scaler_ = _prepare_matrix(
            df,
            self.cols,
        )

        self.model_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            n_init=10,
        ).fit(Xs)

        self.machine_id_ = _machine_cluster_id(
            self.model_.labels_,
            df,
            self.n_clusters,
        )
        return self

    def score(self, df: pd.DataFrame):
        Xs, _, _, _ = _prepare_matrix(
            df,
            self.cols,
            self.medians_,
            self.scaler_,
        )

        distances = self.model_.transform(Xs)

        if self.n_clusters != 2:
            return -distances[:, self.machine_id_]

        other_id = 1 - self.machine_id_
        return (
            distances[:, other_id]
            - distances[:, self.machine_id_]
        )


class FrozenGMM:
    """Two-component full-covariance Gaussian-mixture scorer."""

    def __init__(
        self,
        cols: Sequence[str],
        n_components: int = 2,
        seed: int = 42,
    ):
        self.cols = list(cols)
        self.n_components = int(n_components)
        self.seed = int(seed)

    def fit(self, df: pd.DataFrame):
        Xs, _, self.medians_, self.scaler_ = _prepare_matrix(
            df,
            self.cols,
        )

        self.model_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            random_state=self.seed,
            n_init=3,
        ).fit(Xs)

        labels = self.model_.predict(Xs)

        self.machine_id_ = _machine_cluster_id(
            labels,
            df,
            self.n_components,
        )
        return self

    def score(self, df: pd.DataFrame):
        Xs, _, _, _ = _prepare_matrix(
            df,
            self.cols,
            self.medians_,
            self.scaler_,
        )

        # Weighted component log densities:
        # log pi_k + log N(x | mu_k, Sigma_k).
        logdens = self.model_._estimate_weighted_log_prob(Xs)

        if self.n_components != 2:
            return logdens[:, self.machine_id_]

        other_id = 1 - self.machine_id_
        return (
            logdens[:, self.machine_id_]
            - logdens[:, other_id]
        )


def choose_k(
    fit_df: pd.DataFrame,
    fit_mixture_k_func,
    k_range=range(2, 11),
    rule_frac: float = 1 / 3,
    verbose: bool = True,
):
    """
    Fit candidate custom-mixture models and apply the pre-specified BIC rule.

    Let improvement(K) = BIC(K) - BIC(K+1). Starting from K=2, select the
    first candidate K+1 whose improvement is smaller than `rule_frac` times
    the largest *previous* improvement.

    The first transition cannot trigger stopping because there is no previous
    improvement against which to compare it.
    """
    rows = []

    for k in k_range:
        result = fit_mixture_k_func(
            fit_df,
            K=int(k),
            verbose=False,
        )
        bic = float(result["bic"])
        rows.append({"K": int(k), "BIC": bic})

        if verbose:
            print(f"  K={k:<2d} BIC={bic:.0f}")

    tab = pd.DataFrame(rows)
    tab["dBIC_to_next"] = (
        tab["BIC"]
        - tab["BIC"].shift(-1)
    )

    valid = tab.dropna(
        subset=["dBIC_to_next"]
    )

    previous = []
    chosen = None
    selected_threshold = np.nan

    for _, row in valid.iterrows():
        improvement = float(row["dBIC_to_next"])

        if previous:
            threshold = max(previous) * rule_frac

            if improvement < threshold:
                chosen = int(row["K"] + 1)
                selected_threshold = float(threshold)
                break

        previous.append(improvement)

    if chosen is None:
        chosen = int(tab["K"].iloc[-1])

        if previous:
            selected_threshold = float(
                max(previous) * rule_frac
            )

    tab.attrs["rule_threshold"] = selected_threshold
    return chosen, tab
