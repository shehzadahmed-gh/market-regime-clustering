"""Market regimes from k-means on rolling return distributions.

Each date is described by the distribution of daily log returns over the
window ending that day: nine quantiles, the mean, volatility, skewness,
excess kurtosis and the window's maximum drawdown. Features are standardised
and clustered with k-means.

Cluster numbers from k-means are arbitrary, so regimes are renumbered by
volatility: regime 0 is always the calmest. That also makes labels from
separate fits comparable, which the walk-forward assignment relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
TRADING_DAYS = 252


@dataclass
class RegimeResult:
    observations: pd.DataFrame  # one row per window end: regime, window return and volatility
    features: pd.DataFrame  # the unscaled feature matrix
    cluster_summary: pd.DataFrame  # one row per regime
    silhouette: float
    model: KMeans
    scaler: StandardScaler


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna().rename("log_return")


def _max_drawdown(window_returns: np.ndarray) -> float:
    """Largest peak-to-trough fall within the window, as a simple return (e.g. -0.12)."""
    path = np.concatenate([[0.0], np.cumsum(window_returns)])
    return float(np.exp((path - np.maximum.accumulate(path)).min()) - 1.0)


def rolling_distribution_features(returns: pd.Series, window: int = 126) -> pd.DataFrame:
    """Feature row for every date with a full window behind it."""
    if len(returns) < window:
        raise ValueError(f"Need at least {window} returns for a {window}-day window, got {len(returns)}")
    rolling = returns.rolling(window)
    features = pd.DataFrame({f"q{round(q * 100):02d}": rolling.quantile(q) for q in QUANTILES})
    features["mean"] = rolling.mean()
    features["volatility"] = rolling.std()
    features["skewness"] = rolling.skew()
    features["excess_kurtosis"] = rolling.kurt()
    features["max_drawdown"] = rolling.apply(_max_drawdown, raw=True)
    return features.dropna()


def _ordered_kmeans(scaled: np.ndarray, volatility: np.ndarray, n_clusters: int, random_state: int):
    """Fit k-means and renumber clusters by mean volatility, lowest first."""
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit(scaled)
    order = np.argsort([volatility[model.labels_ == c].mean() for c in range(n_clusters)])
    relabel = np.empty(n_clusters, dtype=int)
    relabel[order] = np.arange(n_clusters)
    return model, relabel


def summarise_regimes(features: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Annualised return and volatility, tail shape and typical length of each regime."""
    runs = (regimes != regimes.shift()).cumsum()
    run_lengths = regimes.groupby(runs).agg(["first", "size"])
    rows = {}
    for regime, group in features.groupby(regimes):
        rows[regime] = {
            "days": len(group),
            "share": len(group) / len(features),
            "ann_return": group["mean"].mean() * TRADING_DAYS,
            "ann_volatility": group["volatility"].mean() * np.sqrt(TRADING_DAYS),
            "q01": group["q01"].mean(),
            "skewness": group["skewness"].mean(),
            "excess_kurtosis": group["excess_kurtosis"].mean(),
            "max_drawdown": group["max_drawdown"].mean(),
            "avg_spell_days": run_lengths.loc[run_lengths["first"] == regime, "size"].mean(),
        }
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "regime"
    return summary


def fit_kmeans_regimes(
    returns: pd.Series,
    window: int = 126,
    n_clusters: int = 4,
    random_state: int = 42,
    feature_columns: list[str] | None = None,
) -> RegimeResult:
    """Cluster every window in the sample at once.

    `feature_columns` restricts clustering to a subset of the features (all
    of them by default); the summary always reports every feature.

    Scaling and clustering use the whole sample, so a date's label depends on
    data after it. That is fine for describing history; for anything that
    trades on the label use `walk_forward_regimes`.
    """
    features = rolling_distribution_features(returns, window)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[feature_columns] if feature_columns else features)
    model, relabel = _ordered_kmeans(scaled, features["volatility"].to_numpy(), n_clusters, random_state)
    labels = relabel[model.labels_]

    regimes = pd.Series(labels, index=features.index, name="regime")
    observations = pd.DataFrame(
        {
            "regime": regimes,
            "window_return": features["mean"] * window,
            "window_volatility": features["volatility"] * np.sqrt(TRADING_DAYS),
        }
    )
    return RegimeResult(
        observations=observations,
        features=features,
        cluster_summary=summarise_regimes(features, regimes),
        silhouette=float(silhouette_score(scaled, labels)),
        model=model,
        scaler=scaler,
    )


def score_cluster_counts(returns: pd.Series, window: int = 126, cluster_counts=range(2, 9)) -> pd.DataFrame:
    """Internal clustering scores for each k. Higher silhouette and
    Calinski-Harabasz are better; lower Davies-Bouldin is better."""
    scaled = StandardScaler().fit_transform(rolling_distribution_features(returns, window))
    rows = {}
    for k in cluster_counts:
        model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(scaled)
        rows[k] = {
            "inertia": model.inertia_,
            "silhouette": silhouette_score(scaled, model.labels_),
            "calinski_harabasz": calinski_harabasz_score(scaled, model.labels_),
            "davies_bouldin": davies_bouldin_score(scaled, model.labels_),
        }
    scores = pd.DataFrame.from_dict(rows, orient="index")
    scores.index.name = "k"
    return scores


def seed_stability(returns: pd.Series, window: int = 126, n_clusters: int = 4, seeds=range(10)) -> float:
    """Mean adjusted Rand index between fits that differ only in the k-means
    seed. 1.0 means every seed finds the same partition."""
    labels = [fit_kmeans_regimes(returns, window, n_clusters, seed).observations["regime"] for seed in seeds]
    return float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]))


def walk_forward_regimes(
    returns: pd.Series,
    window: int = 126,
    n_clusters: int = 4,
    min_history: int = 756,
    refit_every: int = 63,
    random_state: int = 42,
) -> pd.Series:
    """Regime labels that only use data available on each date.

    Every `refit_every` days the scaler and k-means are refitted on all
    feature rows up to that day (at least `min_history` of them), and the
    following block of days is labelled with that fit. Because clusters are
    numbered by volatility in every fit, labels stay comparable across refits.
    """
    features = rolling_distribution_features(returns, window)
    volatility = features["volatility"].to_numpy()
    labels = pd.Series(np.nan, index=features.index, name="regime")
    for start in range(min_history, len(features), refit_every):
        scaler = StandardScaler().fit(features.iloc[:start])
        history = scaler.transform(features.iloc[:start])
        model, relabel = _ordered_kmeans(history, volatility[:start], n_clusters, random_state)
        block = features.iloc[start : start + refit_every]
        labels.iloc[start : start + refit_every] = relabel[model.predict(scaler.transform(block))]
    return labels.dropna().astype(int)
