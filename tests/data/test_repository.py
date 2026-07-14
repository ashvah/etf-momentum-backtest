from dataclasses import replace
from pathlib import Path

import pandas as pd

from etf_momentum_backtest.config import BacktestConfig
from etf_momentum_backtest.data.repository import (
    get_processed_cache_path,
    get_raw_cache_path,
    read_processed_prices,
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


def test_raw_cache_path_contains_provider_and_ticker(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    path = get_raw_cache_path(
        config=config,
        ticker="spy",
    )

    assert path.parent == config.raw_data_dir
    assert path.name == "akshare_sina_qfq_SPY.parquet"


def test_processed_cache_path_contains_experiment_parameters(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    path = get_processed_cache_path(config)

    assert path.parent == config.processed_data_dir
    assert "SPY_QQQ" in path.name
    assert "20240101" in path.name
    assert "20241231" in path.name
    assert "qfq" in path.name


def test_processed_cache_paths_differ_by_date_range(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    other_config = replace(
        config,
        start_date="2020-01-01",
        end_date=None,
    )

    assert get_processed_cache_path(config) != get_processed_cache_path(other_config)


def test_raw_data_round_trip(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    raw_data = pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-03",
            ],
            "open": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1_000, 1_100],
        }
    )

    path = get_raw_cache_path(
        config=config,
        ticker="SPY",
    )

    save_raw_data(
        raw_data=raw_data,
        path=path,
    )

    loaded = read_raw_data(path)

    assert path.exists()
    pd.testing.assert_frame_equal(
        loaded,
        raw_data,
    )


def test_processed_prices_round_trip(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0],
            "QQQ": [200.0, 202.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
            ]
        ),
    )

    path = get_processed_cache_path(config)

    save_processed_prices(
        prices=prices,
        path=path,
    )

    loaded = read_processed_prices(path)

    assert path.exists()
    assert isinstance(
        loaded.index,
        pd.DatetimeIndex,
    )

    pd.testing.assert_frame_equal(
        loaded,
        prices,
    )
