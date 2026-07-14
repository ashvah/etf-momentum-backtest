import numpy as np
import pandas as pd
import pytest

from etf_momentum_backtest.backtest import (
    BacktestResult,
)
from etf_momentum_backtest.metrics import (
    analyze_drawdowns,
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    summarize_performance,
)


def test_total_return() -> None:
    portfolio_value = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ]
        ),
    )

    result = calculate_total_return(portfolio_value)

    assert result == pytest.approx(0.21)


def test_cagr_uses_elapsed_calendar_time() -> None:
    portfolio_value = pd.Series(
        [100.0, 121.0],
        index=pd.to_datetime(
            [
                "2023-01-01",
                "2025-01-01",
            ]
        ),
    )

    elapsed_years = 731 / 365.25
    expected = 1.21 ** (1 / elapsed_years) - 1.0

    assert calculate_cagr(portfolio_value) == pytest.approx(expected)


def test_drawdown_statistics() -> None:
    portfolio_value = pd.Series(
        [
            100.0,
            120.0,
            90.0,
            96.0,
            130.0,
        ],
        index=pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ]
        ),
    )

    statistics = analyze_drawdowns(portfolio_value)

    assert statistics.max_drawdown == pytest.approx(-0.25)

    assert statistics.peak_date == pd.Timestamp("2024-01-02")

    assert statistics.trough_date == pd.Timestamp("2024-01-03")

    assert statistics.recovery_date == pd.Timestamp("2024-01-05")

    assert statistics.max_drawdown_duration_periods == 2


def test_unrecovered_drawdown_has_no_recovery_date() -> None:
    portfolio_value = pd.Series(
        [100.0, 120.0, 90.0, 95.0],
        index=pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ]
        ),
    )

    statistics = analyze_drawdowns(portfolio_value)

    assert statistics.recovery_date is None


def test_annualized_volatility() -> None:
    returns = pd.Series(
        [0.01, -0.01, 0.01, -0.01],
        index=pd.bdate_range(
            "2024-01-01",
            periods=4,
        ),
    )

    expected = returns.std(ddof=1) * np.sqrt(252)

    result = calculate_annualized_volatility(returns)

    assert result == pytest.approx(expected)


def test_sharpe_ratio() -> None:
    returns = pd.Series(
        [0.01, 0.02, -0.01, 0.01],
        index=pd.bdate_range(
            "2024-01-01",
            periods=4,
        ),
    )

    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)

    result = calculate_sharpe_ratio(returns)

    assert result == pytest.approx(expected)


def test_sortino_ratio() -> None:
    returns = pd.Series(
        [0.02, -0.01, 0.01, -0.02],
        index=pd.bdate_range(
            "2024-01-01",
            periods=4,
        ),
    )

    downside = np.minimum(
        returns.to_numpy(),
        0.0,
    )

    downside_deviation = np.sqrt(np.mean(downside**2))

    expected = returns.mean() * 252 / (downside_deviation * np.sqrt(252))

    result = calculate_sortino_ratio(returns)

    assert result == pytest.approx(expected)


def make_backtest_result() -> BacktestResult:
    index = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
        ]
    )

    portfolio_value = pd.Series(
        [
            10_000.0,
            11_000.0,
            9_900.0,
        ],
        index=index,
        name="portfolio_value",
    )

    daily_returns = pd.Series(
        [
            0.0,
            0.10,
            -0.10,
        ],
        index=index,
        name="daily_return",
    )

    actual_weights = pd.DataFrame(
        {
            "SPY": [0.0, 1.0, 1.0],
            "QQQ": [0.0, 0.0, 0.0],
        },
        index=index,
    )

    cash_balance = pd.Series(
        [
            10_000.0,
            0.0,
            0.0,
        ],
        index=index,
        name="cash_balance",
    )

    turnover = pd.Series(
        [0.0, 1.0, 2.0],
        index=index,
        name="turnover",
    )

    transaction_costs = pd.Series(
        [0.0, 10.0, 20.0],
        index=index,
        name="transaction_cost",
    )

    execution_targets = pd.DataFrame(
        {
            "SPY": [1.0, 0.0],
            "QQQ": [0.0, 1.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-03",
                "2024-01-04",
            ]
        ),
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


def test_summarize_performance() -> None:
    result = make_backtest_result()

    metrics = summarize_performance(result)

    assert metrics.initial_value == 10_000.0
    assert metrics.final_value == 9_900.0
    assert metrics.total_return == pytest.approx(-0.01)

    assert metrics.max_drawdown == pytest.approx(-0.10)

    assert metrics.total_turnover == pytest.approx(3.0)

    assert metrics.number_of_rebalances == 2

    assert metrics.average_turnover_per_rebalance == pytest.approx(1.5)

    assert metrics.total_transaction_costs == pytest.approx(30.0)

    assert metrics.transaction_cost_to_initial_capital == pytest.approx(0.003)


def test_inconsistent_daily_returns_are_rejected() -> None:
    result = make_backtest_result()

    invalid_returns = result.daily_returns.copy()
    invalid_returns.iloc[-1] = 0.50

    invalid_result = BacktestResult(
        portfolio_value=result.portfolio_value,
        daily_returns=invalid_returns,
        actual_weights=result.actual_weights,
        cash_balance=result.cash_balance,
        turnover=result.turnover,
        transaction_costs=(result.transaction_costs),
        execution_targets=(result.execution_targets),
    )

    with pytest.raises(
        ValueError,
        match="inconsistent",
    ):
        summarize_performance(invalid_result)
