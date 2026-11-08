"""Price data: real prices from Yahoo Finance, or a simulated series with known regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Daily drift and volatility of each simulated regime (log returns).
SYNTHETIC_REGIMES = {
    "calm": (0.0006, 0.007),
    "normal": (0.0002, 0.012),
    "stressed": (-0.0012, 0.028),
}


def download_prices(ticker: str, start: str = "2005-01-01", end: str | None = None) -> pd.Series:
    """Daily closes adjusted for splits and dividends."""
    import yfinance as yf

    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No price data returned for {ticker!r}")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance returns one column per ticker
        close = close.iloc[:, 0]
    return close.dropna().rename(ticker)


def simulate_regime_prices(
    periods: int = 3000, seed: int = 42, stay_probability: float = 0.99, start_price: float = 100.0
) -> tuple[pd.Series, pd.Series]:
    """Prices whose returns switch between the regimes in SYNTHETIC_REGIMES.

    The regime follows a Markov chain that stays put with `stay_probability`
    each day and otherwise jumps to one of the other regimes, so regimes last
    about 1 / (1 - stay_probability) days on average. Returns (prices, regime
    name per day) so a clustering method can be checked against the truth.
    """
    rng = np.random.default_rng(seed)
    names = list(SYNTHETIC_REGIMES)
    states = np.empty(periods, dtype=int)
    states[0] = 0
    for t in range(1, periods):
        if rng.random() < stay_probability:
            states[t] = states[t - 1]
        else:
            states[t] = rng.choice([s for s in range(len(names)) if s != states[t - 1]])

    drift = np.array([SYNTHETIC_REGIMES[n][0] for n in names])[states]
    vol = np.array([SYNTHETIC_REGIMES[n][1] for n in names])[states]
    log_returns = drift + vol * rng.standard_normal(periods)

    index = pd.bdate_range("2010-01-01", periods=periods)
    prices = pd.Series(start_price * np.exp(np.cumsum(log_returns)), index=index, name="SYNTH")
    regimes = pd.Series(np.array(names)[states], index=index, name="true_regime")
    return prices, regimes


def make_synthetic_prices(periods: int = 3000, seed: int = 42) -> pd.Series:
    """Simulated prices for running the demo without a network connection."""
    prices, _ = simulate_regime_prices(periods=periods, seed=seed)
    return prices
