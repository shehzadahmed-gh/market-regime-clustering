"""Check how well the clustering recovers regimes that are known.

Simulates prices that switch between three regimes (calm, normal, stressed;
see src/data.py), clusters them with k = 3, and compares the labels with the
true regime that covered most of each window, using the adjusted Rand index
(1 = identical partitions, 0 = no better than chance). Runs the full feature
set and volatility alone, for several window lengths and seeds.

    python scripts/validate_on_simulated.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import SYNTHETIC_REGIMES, simulate_regime_prices  # noqa: E402
from src.regime_model import calculate_log_returns, fit_kmeans_regimes  # noqa: E402

WINDOWS = (21, 63, 126)
SEEDS = range(5)


def majority_regime(codes: pd.Series, window: int) -> pd.Series:
    return codes.rolling(window).apply(lambda x: np.bincount(x.astype(int)).argmax(), raw=True)


def main() -> int:
    codes_for = {name: i for i, name in enumerate(SYNTHETIC_REGIMES)}
    rows = []
    for seed in SEEDS:
        prices, truth = simulate_regime_prices(periods=3000, seed=seed)
        returns = calculate_log_returns(prices)
        codes = truth.map(codes_for).reindex(returns.index)
        for window in WINDOWS:
            for name, columns in (("all 14 features", None), ("volatility only", ["volatility"])):
                result = fit_kmeans_regimes(returns, window=window, n_clusters=3, feature_columns=columns)
                target = majority_regime(codes, window).reindex(result.observations.index)
                rows.append(
                    {
                        "seed": seed,
                        "window": window,
                        "features": name,
                        "ari": adjusted_rand_score(target, result.observations["regime"]),
                        "silhouette": result.silhouette,
                    }
                )

    df = pd.DataFrame(rows)
    table = df.groupby(["window", "features"])[["ari", "silhouette"]].mean().round(3).unstack("features")
    print(f"Mean over seeds {list(SEEDS)}; 3,000 simulated days each\n")
    print(table.to_string())
    out = Path("results/simulated_validation.csv")
    out.parent.mkdir(exist_ok=True)
    df.round(4).to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
