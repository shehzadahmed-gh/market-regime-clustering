from datetime import date

import matplotlib.pyplot as plt
import streamlit as st

from src.data import download_prices, make_synthetic_prices
from src.plots import plot_regimes
from src.regime_model import (
    calculate_log_returns,
    fit_kmeans_regimes,
    score_cluster_counts,
)

st.set_page_config(page_title="K-Means Market Regimes", layout="wide")
st.title("K-Means Market Regime Clustering")
st.caption("Clusters rolling empirical return distributions represented by quantiles and distribution statistics.")

with st.sidebar:
    ticker = st.text_input("Ticker", value="SPY").upper().strip()
    # Without explicit limits Streamlit only allows dates within 10 years of the default.
    start = st.date_input("Start date", value=date(2005, 1, 1), min_value=date(1990, 1, 1), max_value=date.today())
    window = st.slider("Rolling window", 40, 252, 126, step=10)
    n_clusters = st.slider("Number of regimes", 2, 8, 4)
    use_synthetic = st.checkbox("Use synthetic data", value=False)
    run = st.button("Run model", type="primary")

if run:
    with st.spinner("Building rolling distributions and fitting K-means..."):
        try:
            if use_synthetic:
                prices = make_synthetic_prices()
                source = "Synthetic data"
            else:
                prices = download_prices(ticker, start=str(start))
                source = "Yahoo Finance via yfinance"

            returns = calculate_log_returns(prices)
            result = fit_kmeans_regimes(
                returns,
                window=window,
                n_clusters=n_clusters,
            )
            scores = score_cluster_counts(
                returns,
                window=window,
                cluster_counts=range(2, 9),
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Data source", source)
    col2.metric("Rolling observations", f"{len(result.observations):,}")
    col3.metric("Silhouette score", f"{result.silhouette:.3f}")

    st.pyplot(plot_regimes(prices, result.observations))
    plt.close("all")

    st.subheader("Regime characteristics")
    st.dataframe(result.cluster_summary.round(4), width="stretch")

    st.subheader("Choosing the cluster count")
    st.line_chart(scores["silhouette"])
    st.dataframe(scores.round(4), width="stretch")

    st.subheader("Latest detected regime")
    latest_date = result.observations.index[-1]
    latest_regime = int(result.observations["regime"].iloc[-1])
    st.write(f"On **{latest_date.date()}**, the model assigns **Regime {latest_regime}**.")

    st.caption(
        "Regimes are numbered from lowest to highest average volatility. Labels are fitted on the "
        "whole sample, so each date's label uses data from after it; see walk_forward_regimes for "
        "labels that only use the past. Not investment advice."
    )
else:
    st.info("Choose settings in the sidebar and press **Run model**.")
