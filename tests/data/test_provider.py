import pandas as pd
import pytest
from requests.exceptions import ConnectionError

from etf_momentum_backtest.data.provider import (
    MarketDataDownloadError,
    download_raw_ticker,
)


def test_download_raw_ticker_returns_provider_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "close": [100.0],
        }
    )

    monkeypatch.setattr(
        "etf_momentum_backtest.data.provider.ak.stock_us_daily",
        lambda **kwargs: expected,
    )

    result = download_raw_ticker(
        ticker="spy",
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_download_retries_after_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    expected = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "close": [100.0],
        }
    )

    def fake_download(
        **kwargs: object,
    ) -> pd.DataFrame:
        nonlocal calls
        calls += 1

        if calls == 1:
            raise ConnectionError("temporary failure")

        return expected

    monkeypatch.setattr(
        "etf_momentum_backtest.data.provider.ak.stock_us_daily",
        fake_download,
    )

    result = download_raw_ticker(
        ticker="SPY",
        max_attempts=2,
        sleep=lambda _: None,
    )

    assert calls == 2
    assert not result.empty


def test_download_fails_after_all_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "etf_momentum_backtest.data.provider.ak.stock_us_daily",
        lambda **kwargs: pd.DataFrame(),
    )

    with pytest.raises(
        MarketDataDownloadError,
        match="after 2 attempts",
    ):
        download_raw_ticker(
            ticker="SPY",
            max_attempts=2,
            sleep=lambda _: None,
        )
