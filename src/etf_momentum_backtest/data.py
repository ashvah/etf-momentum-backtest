from pathlib import Path
import random
import time
import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from etf_momentum_backtest.config import BacktestConfig


def download_prices(
    config: BacktestConfig,
    max_attempts: int = 4,
    base_delay_seconds: float = 2.0,
) -> pd.DataFrame:
    """Download adjusted closing prices with limited retries."""

    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            data = yf.download(
                tickers=list(config.tickers),
                start=config.start_date,
                end=config.end_date,
                auto_adjust=True,
                progress=True,
                threads=False,
                timeout=30,
            )

            if not data.empty:
                prices = extract_close_prices(
                    data=data,
                    tickers=config.tickers,
                )

                return clean_prices(
                    prices=prices,
                    expected_tickers=config.tickers,
                )

        except YFRateLimitError as error:
            last_error = error

        if attempt < max_attempts - 1:
            delay = base_delay_seconds * (2**attempt) + random.uniform(0.0, 1.0)
            time.sleep(delay)

    message = (
        "Yahoo Finance did not return market data after "
        f"{max_attempts} attempts. The data source may be rate limited."
    )

    if last_error is not None:
        raise RuntimeError(message) from last_error

    raise RuntimeError(message)


def extract_close_prices(
    data: pd.DataFrame,
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    """Extract closing prices from yfinance output."""

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("Downloaded data does not contain Close prices.")

        prices = data["Close"].copy()
    else:
        if "Close" not in data.columns:
            raise ValueError("Downloaded data does not contain Close prices.")

        prices = data[["Close"]].copy()
        prices.columns = [tickers[0]]

    return prices


def clean_prices(
    prices: pd.DataFrame,
    expected_tickers: tuple[str, ...],
) -> pd.DataFrame:
    """Normalize and validate the downloaded price matrix."""

    cleaned = prices.copy()

    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.sort_index()
    cleaned = cleaned.loc[~cleaned.index.duplicated(keep="first")]

    cleaned.columns = [str(column).strip().upper() for column in cleaned.columns]

    missing_tickers = set(expected_tickers) - set(cleaned.columns)

    if missing_tickers:
        missing = ", ".join(sorted(missing_tickers))
        raise ValueError(f"Missing downloaded tickers: {missing}")

    cleaned = cleaned.loc[:, list(expected_tickers)]

    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.dropna()

    validate_prices(
        prices=cleaned,
        expected_tickers=expected_tickers,
    )

    return cleaned


def validate_prices(
    prices: pd.DataFrame,
    expected_tickers: tuple[str, ...],
) -> None:
    """Validate the processed price matrix."""

    if prices.empty:
        raise ValueError("Price data is empty.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Price index must be a DatetimeIndex.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price index must be sorted.")

    if prices.index.has_duplicates:
        raise ValueError("Price index contains duplicate dates.")

    if tuple(prices.columns) != expected_tickers:
        raise ValueError("Price columns do not match the configured ticker order.")

    if prices.isna().any().any():
        raise ValueError("Price data contains missing values.")

    if not prices.map(pd.api.types.is_number).all().all():
        raise TypeError("Price data must contain numeric values.")

    if (prices <= 0).any().any():
        raise ValueError("Price data contains non-positive prices.")


def get_price_cache_path(
    config: BacktestConfig,
) -> Path:
    ticker_key = "_".join(config.tickers)

    return config.processed_data_dir / f"prices_{ticker_key}.parquet"


def save_prices(
    prices: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prices.to_parquet(path)


def read_cached_prices(
    path: Path,
) -> pd.DataFrame:
    prices = pd.read_parquet(path)
    prices.index = pd.to_datetime(prices.index)

    return prices


def load_prices(
    config: BacktestConfig,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load processed prices from cache or download them."""

    cache_path = get_price_cache_path(config)

    if cache_path.exists() and not refresh:
        prices = read_cached_prices(cache_path)

        validate_prices(
            prices=prices,
            expected_tickers=config.tickers,
        )

        return prices

    try:
        prices = download_prices(config)
    except RuntimeError:
        # If refreshing fails but a valid cache exists,
        # do not silently replace or delete it.
        if cache_path.exists():
            raise RuntimeError(
                "Fresh data download failed, but an older cache exists. "
                "Run without --refresh-data to use the cached data."
            )

        raise

    save_prices(prices, cache_path)

    return prices
