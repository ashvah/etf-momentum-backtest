from pathlib import Path

import pandas as pd

from etf_momentum_backtest.config import BacktestConfig
from etf_momentum_backtest.data.provider import (
    ADJUSTMENT_METHOD,
    DATA_PROVIDER,
)


PROCESSED_SCHEMA_VERSION = "v1"


def get_raw_cache_path(
    config: BacktestConfig,
    ticker: str,
) -> Path:
    """Return the raw cache path for one ticker."""

    normalized_ticker = ticker.strip().upper()

    filename = f"{DATA_PROVIDER}_{ADJUSTMENT_METHOD}_{normalized_ticker}.parquet"

    return config.raw_data_dir / filename


def get_processed_cache_path(
    config: BacktestConfig,
) -> Path:
    """Return the processed price-matrix cache path."""

    ticker_key = "_".join(config.tickers)
    start_key = config.start_date.replace("-", "")

    end_key = (
        config.end_date.replace("-", "") if config.end_date is not None else "latest"
    )

    filename = (
        f"{PROCESSED_SCHEMA_VERSION}_"
        f"{DATA_PROVIDER}_"
        f"{ADJUSTMENT_METHOD}_"
        f"{ticker_key}_"
        f"{start_key}_"
        f"{end_key}.parquet"
    )

    return config.processed_data_dir / filename


def save_raw_data(
    raw_data: pd.DataFrame,
    path: Path,
) -> None:
    """Save one provider response without its pandas index."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_data.to_parquet(
        path,
        index=False,
    )


def read_raw_data(
    path: Path,
) -> pd.DataFrame:
    """Read one raw provider dataset."""

    return pd.read_parquet(path)


def save_processed_prices(
    prices: pd.DataFrame,
    path: Path,
) -> None:
    """Save the processed price matrix."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prices.to_parquet(
        path,
        index=True,
    )


def read_processed_prices(
    path: Path,
) -> pd.DataFrame:
    """Read a processed price matrix."""

    prices = pd.read_parquet(path)

    prices.index = pd.to_datetime(
        prices.index,
    )

    return prices
