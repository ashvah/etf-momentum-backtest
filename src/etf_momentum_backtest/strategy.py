import pandas as pd

from etf_momentum_backtest.config import BacktestConfig
from etf_momentum_backtest.data.processing import (
    validate_price_matrix,
)


def calculate_momentum(
    prices: pd.DataFrame,
    lookback_days: int,
) -> pd.DataFrame:
    """Calculate trailing simple returns for every asset."""

    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")

    momentum = prices / prices.shift(lookback_days) - 1.0

    return momentum


def get_month_end_signal_dates(
    index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Return the final available trading date of each month."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Signal-date index must be a DatetimeIndex.")

    if index.empty:
        return pd.DatetimeIndex([])

    if not index.is_monotonic_increasing:
        raise ValueError("Signal-date index must be sorted.")

    month_periods = index.to_period("M")

    signal_dates = (
        pd.Series(index=index, data=index).groupby(month_periods).last().to_numpy()
    )

    return pd.DatetimeIndex(signal_dates)


def select_top_k_weights(
    momentum: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Convert signal-date momentum scores into equal target weights."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if top_k > len(momentum.columns):
        raise ValueError("top_k cannot exceed the number of assets.")

    weights = pd.DataFrame(
        0.0,
        index=momentum.index,
        columns=momentum.columns,
    )

    for signal_date, scores in momentum.iterrows():
        if scores.isna().any():
            continue

        ranked_tickers = sorted(
            momentum.columns,
            key=lambda ticker: (
                -scores[ticker],
                ticker,
            ),
        )

        selected = ranked_tickers[:top_k]

        weights.loc[
            signal_date,
            selected,
        ] = 1.0 / top_k

    return weights


def generate_target_weights(
    prices: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Generate monthly momentum target weights."""

    validate_price_matrix(
        prices=prices,
        expected_tickers=config.tickers,
    )

    momentum = calculate_momentum(
        prices=prices,
        lookback_days=config.lookback_days,
    )

    signal_dates = get_month_end_signal_dates(prices.index)

    signal_momentum = momentum.loc[signal_dates]

    # The first lookback period has insufficient history.
    signal_momentum = signal_momentum.dropna(how="any")

    weights = select_top_k_weights(
        momentum=signal_momentum,
        top_k=config.top_k,
    )

    validate_target_weights(
        weights=weights,
        expected_tickers=config.tickers,
        top_k=config.top_k,
    )

    return weights


def validate_target_weights(
    weights: pd.DataFrame,
    expected_tickers: tuple[str, ...],
    top_k: int,
) -> None:
    """Validate signal-date target portfolio weights."""

    if not isinstance(weights, pd.DataFrame):
        raise TypeError("Target weights must be a pandas DataFrame.")

    if weights.empty:
        raise ValueError("Target weights are empty.")

    if not isinstance(
        weights.index,
        pd.DatetimeIndex,
    ):
        raise TypeError("Target-weight index must be a DatetimeIndex.")

    if tuple(weights.columns) != expected_tickers:
        raise ValueError(
            "Target-weight columns do not match the configured ticker order."
        )

    if weights.isna().any().any():
        raise ValueError("Target weights contain missing values.")

    if (weights < 0).any().any():
        raise ValueError("Long-only target weights cannot be negative.")

    row_sums = weights.sum(axis="columns")

    if not row_sums.round(12).eq(1.0).all():
        raise ValueError("Each target-weight row must sum to 1.")

    selected_counts = weights.gt(0).sum(axis="columns")

    if not selected_counts.eq(top_k).all():
        raise ValueError("Each signal must select exactly top_k assets.")
