from dataclasses import dataclass

import pandas as pd

from etf_momentum_backtest.backtest import (
    BacktestResult,
    run_backtest,
)
from etf_momentum_backtest.config import (
    BacktestConfig,
)
from etf_momentum_backtest.data.processing import (
    validate_price_matrix,
)
from etf_momentum_backtest.strategy import (
    validate_target_weights,
)


@dataclass(frozen=True)
class BenchmarkResults:
    """Backtest results for the baseline portfolios."""

    benchmark_ticker: str
    market_buy_and_hold: BacktestResult
    equal_weight_buy_and_hold: BacktestResult
    monthly_equal_weight: BacktestResult

    def as_dict(
        self,
    ) -> dict[str, BacktestResult]:
        """Return benchmark results with display names."""

        return {
            (f"{self.benchmark_ticker} Buy & Hold"): self.market_buy_and_hold,
            ("Equal Weight Buy & Hold"): self.equal_weight_buy_and_hold,
            ("Monthly Equal Weight"): self.monthly_equal_weight,
        }


def validate_signal_dates(
    signal_dates: pd.DatetimeIndex,
    trading_index: pd.DatetimeIndex,
) -> None:
    """Validate benchmark signal dates."""

    if not isinstance(
        signal_dates,
        pd.DatetimeIndex,
    ):
        raise TypeError("signal_dates must be a DatetimeIndex.")

    if signal_dates.empty:
        raise ValueError("signal_dates cannot be empty.")

    if not signal_dates.is_monotonic_increasing:
        raise ValueError("signal_dates must be sorted.")

    if signal_dates.has_duplicates:
        raise ValueError("signal_dates contains duplicate dates.")

    missing_dates = signal_dates.difference(trading_index)

    if not missing_dates.empty:
        formatted_dates = ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates)

        raise ValueError(
            f"Signal dates are missing from the price index: {formatted_dates}"
        )


def generate_single_asset_buy_and_hold_targets(
    prices: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    ticker: str,
) -> pd.DataFrame:
    """Generate one initial full-allocation signal."""

    normalized_ticker = ticker.strip().upper()

    if normalized_ticker not in prices.columns:
        raise ValueError(
            f"Benchmark ticker {normalized_ticker!r} "
            "is not present in the price matrix."
        )

    validate_signal_dates(
        signal_dates=signal_dates,
        trading_index=prices.index,
    )

    first_signal_date = signal_dates[0]

    weights = pd.DataFrame(
        0.0,
        index=pd.DatetimeIndex(
            [first_signal_date],
            name="signal_date",
        ),
        columns=prices.columns,
        dtype="float64",
    )

    weights.loc[
        first_signal_date,
        normalized_ticker,
    ] = 1.0

    validate_target_weights(
        weights=weights,
        expected_tickers=tuple(prices.columns),
    )

    return weights


def generate_equal_weight_buy_and_hold_targets(
    prices: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Generate one initial equal-weight allocation signal."""

    validate_signal_dates(
        signal_dates=signal_dates,
        trading_index=prices.index,
    )

    first_signal_date = signal_dates[0]
    asset_count = len(prices.columns)

    weights = pd.DataFrame(
        1.0 / asset_count,
        index=pd.DatetimeIndex(
            [first_signal_date],
            name="signal_date",
        ),
        columns=prices.columns,
        dtype="float64",
    )

    validate_target_weights(
        weights=weights,
        expected_tickers=tuple(prices.columns),
    )

    return weights


def generate_monthly_equal_weight_targets(
    prices: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Generate equal-weight targets on every signal date."""

    validate_signal_dates(
        signal_dates=signal_dates,
        trading_index=prices.index,
    )

    asset_count = len(prices.columns)

    weights = pd.DataFrame(
        1.0 / asset_count,
        index=pd.DatetimeIndex(
            signal_dates,
            name="signal_date",
        ),
        columns=prices.columns,
        dtype="float64",
    )

    validate_target_weights(
        weights=weights,
        expected_tickers=tuple(prices.columns),
    )

    return weights


def run_benchmarks(
    prices: pd.DataFrame,
    strategy_target_weights: pd.DataFrame,
    config: BacktestConfig,
    benchmark_ticker: str = "TQQQ",
) -> BenchmarkResults:
    """Run baseline portfolios using the strategy's signal calendar.

    All benchmarks wait until the strategy's first available signal
    before investing. This keeps the evaluation period and execution
    timing aligned with the momentum strategy.
    """

    validate_price_matrix(
        prices=prices,
        expected_tickers=config.tickers,
    )

    if strategy_target_weights.empty:
        raise ValueError("strategy_target_weights cannot be empty.")

    signal_dates = pd.DatetimeIndex(strategy_target_weights.index)

    validate_signal_dates(
        signal_dates=signal_dates,
        trading_index=prices.index,
    )

    normalized_benchmark = benchmark_ticker.strip().upper()

    market_targets = generate_single_asset_buy_and_hold_targets(
        prices=prices,
        signal_dates=signal_dates,
        ticker=normalized_benchmark,
    )

    equal_weight_buy_and_hold_targets = generate_equal_weight_buy_and_hold_targets(
        prices=prices,
        signal_dates=signal_dates,
    )

    monthly_equal_weight_targets = generate_monthly_equal_weight_targets(
        prices=prices,
        signal_dates=signal_dates,
    )

    market_result = run_backtest(
        prices=prices,
        target_weights=market_targets,
        config=config,
    )

    equal_weight_buy_and_hold_result = run_backtest(
        prices=prices,
        target_weights=(equal_weight_buy_and_hold_targets),
        config=config,
    )

    monthly_equal_weight_result = run_backtest(
        prices=prices,
        target_weights=(monthly_equal_weight_targets),
        config=config,
    )

    return BenchmarkResults(
        benchmark_ticker=normalized_benchmark,
        market_buy_and_hold=market_result,
        equal_weight_buy_and_hold=(equal_weight_buy_and_hold_result),
        monthly_equal_weight=(monthly_equal_weight_result),
    )
