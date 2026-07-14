from pathlib import Path

import pandas as pd
import pytest

import etf_momentum_backtest.data.pipeline as pipeline
from etf_momentum_backtest.config import BacktestConfig
from etf_momentum_backtest.data.repository import (
    get_processed_cache_path,
    get_raw_cache_path,
    read_raw_data,
    save_processed_prices,
    save_raw_data,
)


def make_config(
    tmp_path: Path,
) -> BacktestConfig:
    return BacktestConfig(
        tickers=("SPY", "QQQ"),
        start_date="2024-01-01",
        end_date="2024-12-31",
        top_k=1,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
    )


def make_raw_data(
    first_price: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-03",
            ],
            "open": [
                first_price - 1,
                first_price,
            ],
            "close": [
                first_price,
                first_price + 1,
            ],
            "volume": [
                1_000,
                1_100,
            ],
        }
    )


def make_processed_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SPY": [100.0, 101.0],
            "QQQ": [200.0, 201.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
            ]
        ),
    )


def test_processed_cache_is_used_without_downloading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    expected = make_processed_prices()

    processed_path = get_processed_cache_path(config)

    save_processed_prices(
        prices=expected,
        path=processed_path,
    )

    def fail_download(
        ticker: str,
    ) -> pd.DataFrame:
        raise AssertionError(f"Unexpected download for {ticker}")

    monkeypatch.setattr(
        pipeline,
        "download_raw_ticker",
        fail_download,
    )

    result = pipeline.load_prices(config)

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_raw_caches_are_used_when_processed_cache_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)

    save_raw_data(
        raw_data=make_raw_data(100.0),
        path=get_raw_cache_path(
            config=config,
            ticker="SPY",
        ),
    )

    save_raw_data(
        raw_data=make_raw_data(200.0),
        path=get_raw_cache_path(
            config=config,
            ticker="QQQ",
        ),
    )

    def fail_download(
        ticker: str,
    ) -> pd.DataFrame:
        raise AssertionError(f"Unexpected download for {ticker}")

    monkeypatch.setattr(
        pipeline,
        "download_raw_ticker",
        fail_download,
    )

    result = pipeline.load_prices(config)

    assert tuple(result.columns) == (
        "SPY",
        "QQQ",
    )
    assert result["SPY"].tolist() == [
        100.0,
        101.0,
    ]
    assert result["QQQ"].tolist() == [
        200.0,
        201.0,
    ]

    assert get_processed_cache_path(config).exists()


def test_only_missing_raw_ticker_is_downloaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)

    save_raw_data(
        raw_data=make_raw_data(100.0),
        path=get_raw_cache_path(
            config=config,
            ticker="SPY",
        ),
    )

    downloaded_tickers: list[str] = []

    def fake_download(
        ticker: str,
    ) -> pd.DataFrame:
        downloaded_tickers.append(ticker)

        if ticker != "QQQ":
            raise AssertionError(f"Unexpected download for {ticker}")

        return make_raw_data(200.0)

    monkeypatch.setattr(
        pipeline,
        "download_raw_ticker",
        fake_download,
    )

    result = pipeline.load_prices(config)

    assert downloaded_tickers == ["QQQ"]
    assert tuple(result.columns) == (
        "SPY",
        "QQQ",
    )

    qqq_raw_path = get_raw_cache_path(
        config=config,
        ticker="QQQ",
    )

    assert qqq_raw_path.exists()


def test_refresh_downloads_every_ticker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)

    # Existing stale raw caches.
    save_raw_data(
        raw_data=make_raw_data(1.0),
        path=get_raw_cache_path(
            config=config,
            ticker="SPY",
        ),
    )

    save_raw_data(
        raw_data=make_raw_data(2.0),
        path=get_raw_cache_path(
            config=config,
            ticker="QQQ",
        ),
    )

    # Existing stale processed cache.
    save_processed_prices(
        prices=pd.DataFrame(
            {
                "SPY": [1.0, 2.0],
                "QQQ": [2.0, 3.0],
            },
            index=pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
        ),
        path=get_processed_cache_path(config),
    )

    downloaded_tickers: list[str] = []

    def fake_download(
        ticker: str,
    ) -> pd.DataFrame:
        downloaded_tickers.append(ticker)

        first_price = {
            "SPY": 100.0,
            "QQQ": 200.0,
        }[ticker]

        return make_raw_data(first_price)

    monkeypatch.setattr(
        pipeline,
        "download_raw_ticker",
        fake_download,
    )

    result = pipeline.load_prices(
        config=config,
        refresh=True,
    )

    assert downloaded_tickers == [
        "SPY",
        "QQQ",
    ]

    assert result["SPY"].tolist() == [
        100.0,
        101.0,
    ]
    assert result["QQQ"].tolist() == [
        200.0,
        201.0,
    ]

    saved_spy_raw = read_raw_data(
        get_raw_cache_path(
            config=config,
            ticker="SPY",
        )
    )

    assert saved_spy_raw["close"].tolist() == [
        100.0,
        101.0,
    ]
