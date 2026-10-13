# Market Regime Clustering with K-Means

identifies market regimes by
clustering **rolling return distributions**, rather than clustering isolated
daily returns.

## What the model does

1. Downloads adjusted SPY prices.
2. Calculates daily log returns.
3. Creates a 126-day rolling window.
4. Represents each window using:
   - 1%, 5%, 10%, 25%, 50%, 75%, 90%, 95% and 99% empirical quantiles
   - mean
   - volatility
   - skewness
   - excess kurtosis
   - maximum drawdown
5. Standardizes the features.
6. Fits K-means.
7. Assigns a regime to each date.
8. Produces regime charts, evaluation scores and cluster summaries.

Using quantiles means Euclidean distance between feature vectors compares the
shapes of empirical distributions. With a dense quantile grid, this is related
to comparing one-dimensional distributions through their quantile functions.

## Project structure

```text
market-regime-kmeans-demo/
├── app.py
├── run_demo.py
├── test_demo.py
├── requirements.txt
├── src/
│   ├── data.py
│   ├── plots.py
│   └── regime_model.py
├── data/
└── outputs/
```

## Run it on macOS in VS Code

### 1. Open the project folder

```bash
cd market-regime-kmeans-demo
code .
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your terminal should now show `(.venv)`.

### 3. Install dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 4. Run the command-line demo

```bash
python3 run_demo.py
```

The output files will appear under `outputs/`.

### 5. Run the dashboard

```bash
python3 -m streamlit run app.py
```

A local browser tab should open automatically.

### 6. Run the test

```bash
python3 -m pip install pytest
python3 -m pytest -q
```

## Suggested learning sequence

### Milestone 1 — Understand returns
Read `calculate_log_returns()` and verify that:

```python
log_return = log(price_today) - log(price_yesterday)
```

### Milestone 2 — Understand rolling distributions
Read `build_distribution_features()`. Each row describes the trailing return
distribution as of that date.

### Milestone 3 — Understand K-means
Read `fit_kmeans_regimes()`. The scaler is fitted before K-means so features
with larger numerical ranges do not dominate Euclidean distance.

### Milestone 4 — Interpret, do not rename blindly
Cluster labels such as 0, 1 and 2 have no intrinsic meaning. Use annualized
return, volatility, skewness, kurtosis and drawdown to describe each cluster.

### Milestone 5 — Evaluate robustness
Try:

- windows: 60, 126 and 252 days
- clusters: 2 through 8
- assets: SPY, QQQ, GLD and TLT
- multiple random seeds
- train/test splits

## Important research limitations

- Overlapping rolling windows are highly dependent.
- K-means assumes spherical clusters under Euclidean distance.
- K-means has no explicit state-transition model.
- Full-sample scaling and fitting introduce look-ahead bias if used for trading.
- A high silhouette score does not prove economic usefulness.
- yfinance is convenient for education, not institutional-quality research.

## Strong extensions

1. Use a denser 99-point quantile grid.
2. Compare K-means with Gaussian mixtures and hidden Markov models.
3. Build an expanding-window, out-of-sample regime assignment process.
4. Add VIX, rates, credit spreads and cross-asset returns.
5. Backtest a regime-conditioned allocation with transaction costs.
6. Measure regime stability using adjusted Rand index across seeds and samples.

## Portfolio description

> Built an unsupervised market-regime detection pipeline that clusters rolling
> empirical return distributions using quantile embeddings and K-means.
> Evaluated cluster separation, characterized regime-specific risk and return,
> and deployed the analysis as an interactive Streamlit dashboard.
