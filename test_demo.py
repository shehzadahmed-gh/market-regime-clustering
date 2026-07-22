from src.data import make_synthetic_prices
from src.regime_model import calculate_log_returns, fit_kmeans_regimes


def test_pipeline_runs():
    prices = make_synthetic_prices(periods=1000)
    returns = calculate_log_returns(prices)
    result = fit_kmeans_regimes(returns, window=60, n_clusters=4)

    assert len(result.observations) > 0
    assert result.observations["regime"].nunique() == 4
    assert -1 <= result.silhouette <= 1
