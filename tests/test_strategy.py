import pandas as pd
import pytest

from etf_momentum_backtest.config import BacktestConfig
from etf_momentum_backtest.strategy import (
    calculate_momentum,
    generate_target_weights,
    get_month_end_signal_dates,
    select_top_k_weights,
)


def test_calculate_momentum() -> None:
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 110.0, 121.0],
            "QQQ": [200.0, 180.0, 198.0],
        },
        index=pd.date_range(
            "2024-01-01",
            periods=3,
            freq="D",
        ),
    )

    momentum = calculate_momentum(
        prices=prices,
        lookback_days=2,
    )

    assert pd.isna(
        momentum.loc[
            "2024-01-01",
            "SPY",
        ]
    )

    assert momentum.loc[
        "2024-01-03",
        "SPY",
    ] == pytest.approx(0.21)

    assert momentum.loc[
        "2024-01-03",
        "QQQ",
    ] == pytest.approx(-0.01)


def test_month_end_signal_dates_use_available_trading_days() -> None:
    index = pd.to_datetime(
        [
            "2024-01-30",
            "2024-01-31",
            "2024-02-27",
            "2024-02-28",
            "2024-02-29",
        ]
    )

    result = get_month_end_signal_dates(index)

    expected = pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-29",
        ]
    )

    pd.testing.assert_index_equal(
        result,
        expected,
    )


def test_select_top_k_weights() -> None:
    momentum = pd.DataFrame(
        {
            "SPY": [0.10],
            "QQQ": [0.20],
            "TLT": [-0.05],
            "GLD": [0.15],
        },
        index=pd.to_datetime(["2024-01-31"]),
    )

    weights = select_top_k_weights(
        momentum=momentum,
        top_k=2,
    )

    assert weights.loc[
        "2024-01-31",
        "QQQ",
    ] == pytest.approx(0.5)

    assert weights.loc[
        "2024-01-31",
        "GLD",
    ] == pytest.approx(0.5)

    assert weights.loc[
        "2024-01-31",
        "SPY",
    ] == pytest.approx(0.0)


def test_generate_target_weights_skips_insufficient_history() -> None:
    index = pd.bdate_range(
        "2024-01-01",
        periods=50,
    )

    prices = pd.DataFrame(
        {
            "SPY": range(100, 150),
            "QQQ": range(200, 250),
        },
        index=index,
        dtype="float64",
    )

    config = BacktestConfig(
        tickers=("SPY", "QQQ"),
        start_date="2024-01-01",
        lookback_days=30,
        top_k=1,
    )

    weights = generate_target_weights(
        prices=prices,
        config=config,
    )

    assert weights.index.min() >= index[30]
