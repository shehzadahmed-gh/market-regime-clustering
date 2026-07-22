from pathlib import Path

from src.data import download_prices, make_synthetic_prices
from src.plots import plot_cluster_scores, plot_regimes
from src.regime_model import calculate_log_returns, fit_kmeans_regimes, score_cluster_counts


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    try:
        prices = download_prices("SPY", start="2005-01-01")
        source = "Yahoo Finance via yfinance"
    except Exception as exc:
        print(f"Download failed ({exc}). Using synthetic data instead.")
        prices = make_synthetic_prices()
        source = "synthetic fallback"

    returns = calculate_log_returns(prices)
    scores = score_cluster_counts(returns, window=126, cluster_counts=range(2, 9))

    # Four clusters are used for a clear demo. In real research, select k using
    # stability, economic interpretability and out-of-sample analysis—not only
    # the maximum silhouette score.
    result = fit_kmeans_regimes(
        returns,
        window=126,
        n_clusters=4,
        random_state=42,
    )

    result.observations.to_csv(OUTPUT_DIR / "regime_observations.csv")
    result.features.to_csv(OUTPUT_DIR / "distribution_features.csv")
    result.cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")
    scores.to_csv(OUTPUT_DIR / "cluster_scores.csv")

    plot_regimes(
        prices,
        result.observations,
        OUTPUT_DIR / "regimes.png",
    )
    plot_cluster_scores(
        scores,
        OUTPUT_DIR / "cluster_scores.png",
    )

    print(f"Data source: {source}")
    print(f"Silhouette score for k=4: {result.silhouette:.3f}")
    print("\nCluster summary:")
    print(result.cluster_summary.round(4))
    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
