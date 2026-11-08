import numpy as np
import pandas as pd
import pytest

from src.data import make_synthetic_prices, simulate_regime_prices
from src.regime_model import (
    _max_drawdown,
    calculate_log_returns,
    fit_kmeans_regimes,
    rolling_distribution_features,
    score_cluster_counts,
    seed_stability,
    walk_forward_regimes,
)


@pytest.fixture(scope="module")
def returns():
    return calculate_log_returns(make_synthetic_prices(periods=1500, seed=1))


def test_pipeline_runs(returns):
    result = fit_kmeans_regimes(returns, window=60, n_clusters=4)
    assert len(result.observations) == len(returns) - 60 + 1
    assert result.observations["regime"].nunique() == 4
    assert -1 <= result.silhouette <= 1
    assert result.cluster_summary["share"].sum() == pytest.approx(1.0)


def test_regimes_are_numbered_by_volatility(returns):
    summary = fit_kmeans_regimes(returns, window=60, n_clusters=4).cluster_summary
    assert list(summary.index) == [0, 1, 2, 3]
    assert summary["ann_volatility"].is_monotonic_increasing


def test_features_match_direct_calculation(returns):
    window = 50
    features = rolling_distribution_features(returns, window)
    last = returns.iloc[-window:]
    row = features.iloc[-1]
    assert row["volatility"] == pytest.approx(last.std())
    assert row["mean"] == pytest.approx(last.mean())
    assert row["q50"] == pytest.approx(last.median())
    quantiles = row[[c for c in features.columns if c.startswith("q")]].to_numpy()
    assert np.all(np.diff(quantiles) >= 0)


def test_max_drawdown_on_known_path():
    # Up 10%, then down to 77% of the peak, then partly back.
    path = np.log([1.0, 1.1, 0.847, 0.95])
    assert _max_drawdown(np.diff(path)) == pytest.approx(0.847 / 1.1 - 1)
    assert _max_drawdown(np.array([0.01, 0.02])) == 0.0


def test_cluster_scores_cover_each_k(returns):
    scores = score_cluster_counts(returns, window=60, cluster_counts=range(2, 5))
    assert list(scores.index) == [2, 3, 4]
    assert {"silhouette", "calinski_harabasz", "davies_bouldin", "inertia"} <= set(scores.columns)


def test_volatility_alone_recovers_simulated_regimes():
    prices, truth = simulate_regime_prices(periods=3000, seed=0)
    returns = calculate_log_returns(prices)
    result = fit_kmeans_regimes(returns, window=63, n_clusters=3, feature_columns=["volatility"])
    # The simulated regimes are calm < normal < stressed by volatility, and
    # labels are numbered by volatility, so they should line up directly.
    true_codes = truth.map({"calm": 0, "normal": 1, "stressed": 2}).reindex(result.observations.index)
    agreement = (result.observations["regime"] == true_codes).mean()
    assert agreement > 0.6


def test_kmeans_is_stable_across_seeds(returns):
    assert seed_stability(returns, window=60, n_clusters=3, seeds=range(3)) > 0.9


def test_walk_forward_labels_do_not_use_future_data(returns):
    kwargs = dict(window=60, n_clusters=3, min_history=400, refit_every=100)
    labels = walk_forward_regimes(returns, **kwargs)

    # Replace everything after a cut-off with very different data. Labels up
    # to the cut-off must not change if they only depend on the past.
    cutoff = labels.index[300]
    changed = returns.copy()
    after = changed.index > cutoff
    changed[after] = np.random.default_rng(9).normal(0, 0.05, after.sum())
    changed_labels = walk_forward_regimes(changed, **kwargs)

    pd.testing.assert_series_equal(labels[labels.index <= cutoff], changed_labels[changed_labels.index <= cutoff])
    assert not labels[labels.index > cutoff].equals(changed_labels[changed_labels.index > cutoff])


def test_too_little_data_raises():
    with pytest.raises(ValueError):
        rolling_distribution_features(pd.Series(np.zeros(10)), window=20)
