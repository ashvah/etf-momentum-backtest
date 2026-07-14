import pandas as pd
import pytest

from etf_momentum_backtest.config import (
    BacktestConfig,
)
from etf_momentum_backtest.data.processing import (
    build_price_matrix,
    extract_close_series,
    validate_price_matrix,
)


def make_raw_prices(
    values: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-03",
            ],
            "close": values,
            "open": values,
        }
    )


def test_extract_close_series() -> None:
    raw_data = make_raw_prices([100.0, 101.0])

    result = extract_close_series(
        raw_data=raw_data,
        ticker="SPY",
        start_date="2024-01-01",
        end_date=None,
    )

    assert result.name == "SPY"
    assert result.tolist() == [
        100.0,
        101.0,
    ]
    assert result.index.is_monotonic_increasing


def test_build_price_matrix() -> None:
    config = BacktestConfig(
        tickers=("SPY", "QQQ"),
        start_date="2024-01-01",
        top_k=1,
    )

    raw_data = {
        "SPY": make_raw_prices([100.0, 101.0]),
        "QQQ": make_raw_prices([200.0, 202.0]),
    }

    prices = build_price_matrix(
        raw_data=raw_data,
        config=config,
    )

    assert tuple(prices.columns) == (
        "SPY",
        "QQQ",
    )
    assert len(prices) == 2


def test_non_positive_processed_prices_are_rejected() -> None:
    prices = pd.DataFrame(
        {
            "SPY": [-1.0, 1.0],
            "QQQ": [100.0, 101.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        validate_price_matrix(
            prices=prices,
            expected_tickers=(
                "SPY",
                "QQQ",
            ),
        )
