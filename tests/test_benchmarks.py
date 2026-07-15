import pandas as pd
import pytest

from etf_momentum_backtest.benchmarks import (
    generate_equal_weight_buy_and_hold_targets,
    generate_monthly_equal_weight_targets,
    generate_single_asset_buy_and_hold_targets,
    run_benchmarks,
)
from etf_momentum_backtest.config import (
    BacktestConfig,
)


def make_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SPY": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
            ],
            "QQQ": [
                100.0,
                100.0,
                102.0,
                101.0,
                103.0,
                104.0,
            ],
            "GLD": [
                100.0,
                101.0,
                100.0,
                102.0,
                101.0,
                103.0,
            ],
        },
        index=pd.to_datetime(
            [
                "2024-01-30",
                "2024-01-31",
                "2024-02-01",
                "2024-02-02",
                "2024-02-05",
                "2024-02-06",
            ]
        ),
        dtype="float64",
    )


def make_signal_dates() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [
            "2024-01-31",
            "2024-02-05",
        ]
    )


def make_config() -> BacktestConfig:
    return BacktestConfig(
        tickers=(
            "SPY",
            "QQQ",
            "GLD",
        ),
        start_date="2024-01-30",
        end_date="2024-02-06",
        lookback_days=1,
        top_k=1,
        transaction_cost_rate=0.0,
        initial_capital=10_000.0,
    )


def make_strategy_targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SPY": [1.0, 0.0],
            "QQQ": [0.0, 1.0],
            "GLD": [0.0, 0.0],
        },
        index=make_signal_dates(),
        dtype="float64",
    )


def test_single_asset_buy_and_hold_has_one_signal() -> None:
    weights = generate_single_asset_buy_and_hold_targets(
        prices=make_prices(),
        signal_dates=make_signal_dates(),
        ticker="spy",
    )

    assert len(weights) == 1

    assert weights.index[0] == pd.Timestamp("2024-01-31")

    assert weights.loc[
        "2024-01-31",
        "SPY",
    ] == pytest.approx(1.0)

    assert weights.loc[
        "2024-01-31",
        "QQQ",
    ] == pytest.approx(0.0)


def test_equal_weight_buy_and_hold_has_one_signal() -> None:
    weights = generate_equal_weight_buy_and_hold_targets(
        prices=make_prices(),
        signal_dates=make_signal_dates(),
    )

    assert len(weights) == 1

    assert weights.loc["2024-01-31"].tolist() == pytest.approx(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ]
    )


def test_monthly_equal_weight_uses_every_signal_date() -> None:
    weights = generate_monthly_equal_weight_targets(
        prices=make_prices(),
        signal_dates=make_signal_dates(),
    )

    assert weights.index.equals(make_signal_dates())

    assert len(weights) == 2

    assert weights.iloc[0].tolist() == pytest.approx(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ]
    )

    assert weights.iloc[1].tolist() == pytest.approx(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ]
    )


def test_unknown_benchmark_ticker_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="is not present",
    ):
        generate_single_asset_buy_and_hold_targets(
            prices=make_prices(),
            signal_dates=make_signal_dates(),
            ticker="TLT",
        )


def test_run_benchmarks_uses_common_signal_calendar() -> None:
    results = run_benchmarks(
        prices=make_prices(),
        strategy_target_weights=(make_strategy_targets()),
        config=make_config(),
        benchmark_ticker="SPY",
    )

    market_execution_targets = results.market_buy_and_hold.execution_targets

    equal_weight_execution_targets = results.equal_weight_buy_and_hold.execution_targets

    monthly_execution_targets = results.monthly_equal_weight.execution_targets

    assert len(market_execution_targets) == 1
    assert len(equal_weight_execution_targets) == 1
    assert len(monthly_execution_targets) == 2

    # The first signal on Jan 31 executes on Feb 1.
    assert market_execution_targets.index[0] == (pd.Timestamp("2024-02-01"))

    assert market_execution_targets.loc[
        "2024-02-01",
        "SPY",
    ] == pytest.approx(1.0)


def test_market_benchmark_is_in_cash_before_execution() -> None:
    results = run_benchmarks(
        prices=make_prices(),
        strategy_target_weights=(make_strategy_targets()),
        config=make_config(),
    )

    result = results.market_buy_and_hold

    assert result.portfolio_value.loc["2024-01-31"] == pytest.approx(10_000.0)

    assert result.cash_balance.loc["2024-01-31"] == pytest.approx(10_000.0)

    assert result.actual_weights.loc[
        "2024-01-31",
        "SPY",
    ] == pytest.approx(0.0)

    assert result.actual_weights.loc[
        "2024-02-01",
        "SPY",
    ] == pytest.approx(1.0)


def test_monthly_equal_weight_rebalances_on_each_execution_date() -> None:
    results = run_benchmarks(
        prices=make_prices(),
        strategy_target_weights=(make_strategy_targets()),
        config=make_config(),
    )

    result = results.monthly_equal_weight

    first_execution = pd.Timestamp("2024-02-01")

    second_execution = pd.Timestamp("2024-02-06")

    assert result.actual_weights.loc[first_execution].tolist() == pytest.approx(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ]
    )

    assert result.actual_weights.loc[second_execution].tolist() == pytest.approx(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ]
    )
