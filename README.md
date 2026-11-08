# Market Regime Clustering

[![CI](https://github.com/shehzadahmed-gh/market-regime-clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/shehzadahmed-gh/market-regime-clustering/actions/workflows/ci.yml)

Finds market regimes in SPY by clustering the distribution of daily returns over a rolling window, rather than individual days' returns. Each date is described by the shape of the last 126 days of returns, those descriptions are clustered with k-means, and each cluster is summarised by its annualised return, volatility, tail behaviour and typical duration. There's a command-line run that saves tables and charts, and a Streamlit app.

## Method

1. Download adjusted SPY closes from Yahoo Finance and take daily log returns.
2. For every date with 126 days of history, describe that window with 14 numbers: the 1st, 5th, 10th, 25th, 50th, 75th, 90th, 95th and 99th percentiles, the mean, volatility, skewness, excess kurtosis and the maximum drawdown inside the window. The percentiles are a coarse version of the window's quantile function, so Euclidean distance between two rows roughly compares the shapes of the two return distributions.
3. Standardise the 14 features and fit k-means (k = 4 by default, 20 initialisations).
4. Renumber the clusters by average volatility, so regime 0 is always the calmest. k-means numbers clusters arbitrarily, and this makes labels comparable between runs.
5. Summarise each regime and score k = 2 to 8 with silhouette, Calinski-Harabasz and Davies-Bouldin.

`fit_kmeans_regimes` fits the whole sample at once, so a date's label depends on later data. That is fine for describing history but not for trading on. `walk_forward_regimes` refits every 63 trading days using only data up to that day (after a three-year warm-up) and labels the next block with that fit; a test replaces all returns after a cut-off date and checks that earlier labels don't change.

## Does it find real regimes?

There is no ground truth for SPY, so `scripts/validate_on_simulated.py` checks the method on simulated prices with three known regimes (calm, normal and stressed, with daily volatility of 0.7%, 1.2% and 2.8% and an average spell of about 100 days). It clusters with k = 3 and compares the labels with the true regime that covered most of each window, using the adjusted Rand index (ARI: 1 means identical partitions, 0 means no better than chance). Means over five seeds:

| Window | ARI, all 14 features | ARI, volatility only | Silhouette, all 14 features |
|---|---|---|---|
| 21 days | 0.448 | 0.616 | 0.461 |
| 63 days | 0.470 | 0.631 | 0.405 |
| 126 days | 0.352 | 0.390 | 0.432 |

Two things stand out:

- Clustering on volatility alone recovers the simulated regimes better than all 14 features. These regimes differ almost entirely in volatility, so after standardisation the other 13 features mostly add noise with equal weight. Real returns have fat tails and skew that change between regimes, so the full feature set may do better there, but this result means it has to earn its place.
- Silhouette doesn't track accuracy. The 126-day window scores a higher silhouette than the 63-day window while matching the true regimes less well: long windows blur short regimes into smooth, well-separated clusters that are wrong. So I don't pick k or the window by silhouette alone.

Different k-means seeds give almost the same partition (mean pairwise ARI of 0.95 in the demo run), so the gap to the true regimes comes from the features and the method, not from random initialisation.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_demo.py                          # SPY since 2005; tables and charts in outputs/
python -m streamlit run app.py              # interactive version
python scripts/validate_on_simulated.py     # the table above
pytest -q
```

If the Yahoo Finance download fails, `run_demo.py` falls back to simulated prices and says so. `outputs/` gets the regime label for every date (full-sample and walk-forward), the feature matrix, the regime summary, cluster-count scores, and two charts.

## Layout

```text
src/data.py              price download and the regime-switching simulator
src/regime_model.py      features, k-means, summaries, cluster scores, walk-forward labels
src/plots.py             regime and cluster-score charts
run_demo.py              command-line run
app.py                   Streamlit app
scripts/                 validation on simulated regimes
tests/                   unit tests, including the no-look-ahead check
```

## Limitations

- Overlapping windows share most of their data, so neighbouring rows are far from independent and clustering scores look better than they should.
- k-means assumes roughly spherical clusters of similar size, and has no notion of time or of how likely one regime is to follow another. A Gaussian mixture or a hidden Markov model would address both.
- The labels describe the past distribution of returns. Nothing here tests whether knowing the current regime helps predict future returns or risk; that would need an out-of-sample backtest using the walk-forward labels, with transaction costs.
- SPY alone; adding VIX, rates and credit spreads would give the clustering more to work with.
