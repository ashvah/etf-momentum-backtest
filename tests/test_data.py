import pandas as pd
from typing import Any
import pytest

from etf_momentum_backtest.data import (
    clean_prices,
    validate_prices,
    resolve_akshare_symbols,
    download_single_ticker,
)


def test_akshare_symbols_are_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_universe = pd.DataFrame(
        {
            "代码": [
                "107.SPY",
                "105.QQQ",
                "106.GLD",
            ]
        }
    )

    monkeypatch.setattr(
        "etf_momentum_backtest.data.ak.stock_us_spot_em",
        lambda: fake_universe,
    )

    result = resolve_akshare_symbols(("SPY", "QQQ", "GLD"))

    assert result == {
        "SPY": "107.SPY",
        "QQQ": "105.QQQ",
        "GLD": "106.GLD",
    }


def make_valid_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0],
            "QQQ": [200.0, 202.0, 204.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ]
        ),
    )


def test_valid_prices_are_accepted() -> None:
    prices = make_valid_prices()

    validate_prices(
        prices,
        expected_tickers=("SPY", "QQQ"),
    )


def test_prices_are_sorted() -> None:
    prices = make_valid_prices().sort_index(ascending=False)

    cleaned = clean_prices(
        prices,
        expected_tickers=("SPY", "QQQ"),
    )

    assert cleaned.index.is_monotonic_increasing


def test_duplicate_dates_are_removed() -> None:
    prices = make_valid_prices()
    prices = pd.concat([prices, prices.iloc[[0]]])

    cleaned = clean_prices(
        prices,
        expected_tickers=("SPY", "QQQ"),
    )

    assert not cleaned.index.has_duplicates


def test_non_positive_prices_are_rejected() -> None:
    prices = make_valid_prices()
    prices.loc[
        pd.Timestamp("2024-01-03"),
        "SPY",
    ] = 0.0

    with pytest.raises(
        ValueError,
        match="non-positive",
    ):
        validate_prices(
            prices,
            expected_tickers=("SPY", "QQQ"),
        )


def test_missing_ticker_is_rejected() -> None:
    prices = make_valid_prices()[["SPY"]]

    with pytest.raises(
        ValueError,
        match="Missing downloaded tickers",
    ):
        clean_prices(
            prices,
            expected_tickers=("SPY", "QQQ"),
        )


def test_single_ticker_download_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_data = pd.DataFrame(
        {
            "日期": [
                "2024-01-03",
                "2024-01-02",
            ],
            "收盘": [
                101.0,
                100.0,
            ],
        }
    )

    def fake_stock_us_hist(
        **kwargs: Any,
    ) -> pd.DataFrame:
        return fake_data

    monkeypatch.setattr(
        "etf_momentum_backtest.data.ak.stock_us_hist",
        fake_stock_us_hist,
    )

    prices = download_single_ticker(
        ticker="SPY",
        provider_symbol="107.SPY",
        start_date="20240101",
        end_date="20240131",
    )

    assert prices.name == "SPY"
    assert prices.index.is_monotonic_increasing
    assert prices.tolist() == [100.0, 101.0]


def test_empty_akshare_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "etf_momentum_backtest.data.ak.stock_us_hist",
        lambda **kwargs: pd.DataFrame(),
    )

    with pytest.raises(
        RuntimeError,
        match="returned no data",
    ):
        download_single_ticker(
            ticker="SPY",
            provider_symbol="107.SPY",
            start_date="20240101",
            end_date="20240131",
        )
