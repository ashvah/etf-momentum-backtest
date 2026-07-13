import pandas as pd
import pytest

from etf_momentum_backtest.config import BacktestConfig
from etf_momentum_backtest.data import (
    clean_prices,
    validate_prices,
    download_prices,
)


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


def test_empty_download_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(**kwargs: object) -> pd.DataFrame:
        return pd.DataFrame()

    monkeypatch.setattr(
        "etf_momentum_backtest.data.yf.download",
        fake_download,
    )

    config = BacktestConfig(
        tickers=("SPY", "IWM", "GLD"),
        top_k=2,
    )

    with pytest.raises(
        RuntimeError,
        match="did not return market data",
    ):
        download_prices(
            config=config,
            max_attempts=1,
        )
