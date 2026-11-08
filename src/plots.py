from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_regimes(prices: pd.Series, observations: pd.DataFrame, path: str | Path | None = None):
    """Price (log scale) and rolling volatility, coloured by regime."""
    regimes = observations["regime"]
    aligned = prices.reindex(regimes.index)
    cmap = plt.get_cmap("viridis", regimes.nunique())

    fig, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[2, 1])
    ax_price.plot(prices.index, prices.values, color="lightgrey", linewidth=1, zorder=1)
    for i, regime in enumerate(sorted(regimes.unique())):
        mask = regimes == regime
        ax_price.scatter(aligned.index[mask], aligned[mask], s=3, color=cmap(i), label=f"Regime {regime}", zorder=2)
        ax_vol.scatter(
            observations.index[mask], observations.loc[mask, "window_volatility"], s=3, color=cmap(i), zorder=2
        )
    ax_price.set_yscale("log")
    ax_price.set_ylabel("Price (log scale)")
    ax_price.legend(markerscale=4, loc="upper left")
    ax_vol.set_ylabel("Rolling volatility (annualised)")
    ax_price.set_title("Regimes from k-means on rolling return distributions (0 = lowest volatility)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_cluster_scores(scores: pd.DataFrame, path: str | Path | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, column, note in zip(
        axes,
        ["silhouette", "calinski_harabasz", "davies_bouldin"],
        ["higher is better", "higher is better", "lower is better"],
        strict=True,
    ):
        ax.plot(scores.index, scores[column], marker="o")
        ax.set_title(f"{column} ({note})")
        ax.set_xlabel("k")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
