import pandas as pd
import pytest

from etf_momentum_backtest.backtest import (
    calculate_turnover,
    map_signals_to_execution_dates,
    run_backtest,
)
from etf_momentum_backtest.config import (
    BacktestConfig,
)


def make_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SPY": [
                100.0,
                100.0,
                110.0,
                121.0,
            ],
            "QQQ": [
                100.0,
                100.0,
                100.0,
                100.0,
            ],
        },
        index=pd.to_datetime(
            [
                "2024-01-30",
                "2024-01-31",
                "2024-02-01",
                "2024-02-02",
            ]
        ),
    )


def make_config(
    transaction_cost_rate: float = 0.0,
) -> BacktestConfig:
    return BacktestConfig(
        tickers=("SPY", "QQQ"),
        start_date="2024-01-30",
        end_date="2024-02-02",
        lookback_days=1,
        top_k=1,
        transaction_cost_rate=transaction_cost_rate,
        initial_capital=10_000.0,
    )


def test_signal_is_mapped_to_next_trading_date() -> None:
    weights = pd.DataFrame(
        {
            "SPY": [1.0],
            "QQQ": [0.0],
        },
        index=pd.to_datetime(["2024-01-31"]),
    )

    execution = map_signals_to_execution_dates(
        target_weights=weights,
        trading_index=make_prices().index,
    )

    assert execution.index.tolist() == [pd.Timestamp("2024-02-01")]


def test_full_rotation_has_turnover_two() -> None:
    current = pd.Series(
        {
            "SPY": 1.0,
            "QQQ": 0.0,
        }
    )

    target = pd.Series(
        {
            "SPY": 0.0,
            "QQQ": 1.0,
        }
    )

    assert calculate_turnover(
        current_weights=current,
        target_weights=target,
    ) == pytest.approx(2.0)


def test_first_investment_has_turnover_one() -> None:
    current = pd.Series(
        {
            "SPY": 0.0,
            "QQQ": 0.0,
        }
    )

    target = pd.Series(
        {
            "SPY": 1.0,
            "QQQ": 0.0,
        }
    )

    assert calculate_turnover(
        current_weights=current,
        target_weights=target,
    ) == pytest.approx(1.0)


def test_new_weights_do_not_receive_execution_day_return() -> None:
    prices = make_prices()
    config = make_config()

    target_weights = pd.DataFrame(
        {
            "SPY": [1.0],
            "QQQ": [0.0],
        },
        index=pd.to_datetime(["2024-01-31"]),
    )

    result = run_backtest(
        prices=prices,
        target_weights=target_weights,
        config=config,
    )

    # The signal is executed at the 2024-02-01 close.
    # The portfolio was cash before that close, so it does not
    # receive SPY's return from Jan 31 to Feb 1.
    assert result.portfolio_value.loc["2024-02-01"] == pytest.approx(10_000.0)

    # It receives SPY's return from Feb 1 to Feb 2.
    assert result.portfolio_value.loc["2024-02-02"] == pytest.approx(11_000.0)


def test_transaction_cost_is_deducted_on_execution_date() -> None:
    prices = make_prices()
    config = make_config(
        transaction_cost_rate=0.001,
    )

    target_weights = pd.DataFrame(
        {
            "SPY": [1.0],
            "QQQ": [0.0],
        },
        index=pd.to_datetime(["2024-01-31"]),
    )

    result = run_backtest(
        prices=prices,
        target_weights=target_weights,
        config=config,
    )

    # Initial deployment turnover is 1.
    # Cost = 10,000 × 1 × 0.001 = 10.
    assert result.turnover.loc["2024-02-01"] == pytest.approx(1.0)

    assert result.transaction_costs.loc["2024-02-01"] == pytest.approx(10.0)

    assert result.portfolio_value.loc["2024-02-01"] == pytest.approx(9_990.0)


def test_weights_drift_between_rebalances() -> None:
    prices = pd.DataFrame(
        {
            "SPY": [
                100.0,
                100.0,
                200.0,
            ],
            "QQQ": [
                100.0,
                100.0,
                100.0,
            ],
        },
        index=pd.to_datetime(
            [
                "2024-01-30",
                "2024-01-31",
                "2024-02-01",
            ]
        ),
    )

    config = BacktestConfig(
        tickers=("SPY", "QQQ"),
        start_date="2024-01-30",
        end_date="2024-02-01",
        lookback_days=1,
        top_k=2,
        transaction_cost_rate=0.0,
        initial_capital=10_000.0,
    )

    # Signal on Jan 30 executes on Jan 31.
    target_weights = pd.DataFrame(
        {
            "SPY": [0.5],
            "QQQ": [0.5],
        },
        index=pd.to_datetime(["2024-01-30"]),
    )

    result = run_backtest(
        prices=prices,
        target_weights=target_weights,
        config=config,
    )

    # From Jan 31 to Feb 1, SPY doubles and QQQ is unchanged.
    # Holdings become 10,000 and 5,000.
    assert result.actual_weights.loc[
        "2024-02-01",
        "SPY",
    ] == pytest.approx(2 / 3)

    assert result.actual_weights.loc[
        "2024-02-01",
        "QQQ",
    ] == pytest.approx(1 / 3)
