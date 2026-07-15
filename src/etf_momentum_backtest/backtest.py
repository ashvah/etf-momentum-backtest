from dataclasses import dataclass

import pandas as pd

from etf_momentum_backtest.config import BacktestConfig
from etf_momentum_backtest.data.processing import (
    validate_price_matrix,
)
from etf_momentum_backtest.strategy import (
    validate_target_weights,
)


@dataclass(frozen=True)
class BacktestResult:
    """Outputs produced by the portfolio simulation."""

    portfolio_value: pd.Series
    daily_returns: pd.Series
    actual_weights: pd.DataFrame
    cash_balance: pd.Series
    turnover: pd.Series
    transaction_costs: pd.Series
    execution_targets: pd.DataFrame


def map_signals_to_execution_dates(
    target_weights: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Move each signal to the following available trading date.

    Target weights are indexed by signal dates. A signal generated
    after the close on date t is executed at the close of the next
    available trading date.
    """

    if not isinstance(trading_index, pd.DatetimeIndex):
        raise TypeError("trading_index must be a DatetimeIndex.")

    if trading_index.empty:
        raise ValueError("trading_index cannot be empty.")

    if not trading_index.is_monotonic_increasing:
        raise ValueError("trading_index must be sorted.")

    if trading_index.has_duplicates:
        raise ValueError("trading_index contains duplicate dates.")

    if not isinstance(
        target_weights.index,
        pd.DatetimeIndex,
    ):
        raise TypeError("Target-weight index must be a DatetimeIndex.")

    missing_signal_dates = target_weights.index.difference(trading_index)

    if not missing_signal_dates.empty:
        raise ValueError(
            "Target weights contain signal dates that are "
            "not present in the trading index."
        )

    execution_dates: list[pd.Timestamp] = []
    executable_rows: list[pd.Series] = []

    for signal_date, weights in target_weights.iterrows():
        execution_position = trading_index.searchsorted(
            signal_date,
            side="right",
        )

        # A signal on the final available trading date cannot be
        # executed because there is no following trading date.
        if execution_position >= len(trading_index):
            continue

        execution_date = trading_index[execution_position]

        execution_dates.append(execution_date)
        executable_rows.append(weights)

    if not executable_rows:
        raise ValueError(
            "No target weights can be executed within the available price range."
        )

    execution_targets = pd.DataFrame(
        executable_rows,
        index=pd.DatetimeIndex(
            execution_dates,
            name="execution_date",
        ),
        columns=target_weights.columns,
        dtype="float64",
    )

    if execution_targets.index.has_duplicates:
        raise ValueError("Multiple signals map to the same execution date.")

    return execution_targets


def calculate_turnover(
    current_weights: pd.Series,
    target_weights: pd.Series,
) -> float:
    """Calculate total traded notional as a share of capital.

    Turnover is the sum of absolute changes in risky-asset weights.
    """

    if not current_weights.index.equals(target_weights.index):
        raise ValueError("Current and target weights must have matching asset indexes.")

    return float((target_weights - current_weights).abs().sum())


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    """Simulate a long-only monthly momentum portfolio."""

    validate_price_matrix(
        prices=prices,
        expected_tickers=config.tickers,
    )

    validate_target_weights(
        weights=target_weights,
        expected_tickers=config.tickers,
    )

    execution_targets = map_signals_to_execution_dates(
        target_weights=target_weights,
        trading_index=prices.index,
    )

    assets = list(config.tickers)
    dates = prices.index

    portfolio_value = pd.Series(
        index=dates,
        dtype="float64",
        name="portfolio_value",
    )

    daily_returns = pd.Series(
        0.0,
        index=dates,
        dtype="float64",
        name="daily_return",
    )

    cash_balance = pd.Series(
        index=dates,
        dtype="float64",
        name="cash_balance",
    )

    turnover = pd.Series(
        0.0,
        index=dates,
        dtype="float64",
        name="turnover",
    )

    transaction_costs = pd.Series(
        0.0,
        index=dates,
        dtype="float64",
        name="transaction_cost",
    )

    actual_weights = pd.DataFrame(
        0.0,
        index=dates,
        columns=assets,
        dtype="float64",
    )

    # Dollar value invested in each ETF.
    holdings = pd.Series(
        0.0,
        index=assets,
        dtype="float64",
    )

    cash = float(config.initial_capital)
    previous_end_value = float(config.initial_capital)

    for position, current_date in enumerate(dates):
        # First apply the close-to-close price movement of the
        # holdings that existed at the previous close.
        if position > 0:
            previous_date = dates[position - 1]

            price_relatives = prices.loc[current_date] / prices.loc[previous_date]

            holdings = holdings * price_relatives

        pretrade_value = float(holdings.sum() + cash)

        if pretrade_value <= 0:
            raise RuntimeError(
                f"Portfolio value became non-positive on {current_date.date()}."
            )

        current_risky_weights = holdings / pretrade_value

        if current_date in execution_targets.index:
            target = execution_targets.loc[current_date]

            current_turnover = calculate_turnover(
                current_weights=current_risky_weights,
                target_weights=target,
            )

            transaction_cost = (
                pretrade_value * current_turnover * config.transaction_cost_rate
            )

            if transaction_cost >= pretrade_value:
                raise RuntimeError(
                    "Transaction costs exhausted the "
                    f"portfolio on {current_date.date()}."
                )

            post_cost_value = pretrade_value - transaction_cost

            # Rebalance exactly to the requested target weights
            # after deducting transaction costs.
            holdings = target * post_cost_value

            cash = float(post_cost_value - holdings.sum())

            turnover.loc[current_date] = current_turnover

            transaction_costs.loc[current_date] = transaction_cost

        end_value = float(holdings.sum() + cash)

        portfolio_value.loc[current_date] = end_value

        cash_balance.loc[current_date] = cash

        actual_weights.loc[current_date] = holdings / end_value

        if position == 0:
            daily_returns.loc[current_date] = 0.0
        else:
            daily_returns.loc[current_date] = end_value / previous_end_value - 1.0

        previous_end_value = end_value

    validate_backtest_result(
        result=BacktestResult(
            portfolio_value=portfolio_value,
            daily_returns=daily_returns,
            actual_weights=actual_weights,
            cash_balance=cash_balance,
            turnover=turnover,
            transaction_costs=transaction_costs,
            execution_targets=execution_targets,
        ),
        config=config,
    )

    return BacktestResult(
        portfolio_value=portfolio_value,
        daily_returns=daily_returns,
        actual_weights=actual_weights,
        cash_balance=cash_balance,
        turnover=turnover,
        transaction_costs=transaction_costs,
        execution_targets=execution_targets,
    )


def validate_backtest_result(
    result: BacktestResult,
    config: BacktestConfig,
) -> None:
    """Validate internal consistency of a completed backtest."""

    if result.portfolio_value.empty:
        raise ValueError("Backtest portfolio value is empty.")

    if result.portfolio_value.isna().any():
        raise ValueError("Portfolio value contains missing values.")

    if (result.portfolio_value <= 0).any():
        raise ValueError("Portfolio value contains non-positive values.")

    if result.daily_returns.isna().any():
        raise ValueError("Daily returns contain missing values.")

    if result.actual_weights.isna().any().any():
        raise ValueError("Actual weights contain missing values.")

    if (result.actual_weights < -1e-12).any().any():
        raise ValueError("Actual weights contain negative values.")

    if (result.cash_balance < -1e-8).any():
        raise ValueError("Cash balance became materially negative.")

    if (result.turnover < 0).any():
        raise ValueError("Turnover contains negative values.")

    if (result.transaction_costs < 0).any():
        raise ValueError("Transaction costs contain negative values.")

    expected_columns = tuple(config.tickers)

    if tuple(result.actual_weights.columns) != expected_columns:
        raise ValueError(
            "Actual-weight columns do not match the configured ticker order."
        )
