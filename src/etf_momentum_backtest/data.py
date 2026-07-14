from datetime import date
from pathlib import Path
import time
import akshare as ak
import pandas as pd

from etf_momentum_backtest.config import BacktestConfig
from requests.exceptions import (
    ConnectionError as RequestsConnectionError,
)
from requests.exceptions import ProxyError, Timeout


def format_akshare_date(value: str | None) -> str:
    """Convert an ISO date into AKShare's YYYYMMDD format."""

    if value is None:
        return date.today().strftime("%Y%m%d")

    return date.fromisoformat(value).strftime("%Y%m%d")


def resolve_akshare_symbols(
    tickers: tuple[str, ...],
) -> dict[str, str]:
    """Map ordinary tickers to Eastmoney symbols used by AKShare."""

    universe = ak.stock_us_spot_em()

    if universe.empty:
        raise RuntimeError("AKShare returned an empty U.S. ticker universe.")

    if "代码" not in universe.columns:
        raise RuntimeError("AKShare ticker universe does not contain the 代码 column.")

    codes = universe["代码"].astype(str).str.strip()

    lookup = pd.DataFrame(
        {
            "provider_symbol": codes,
            "ticker": (codes.str.rsplit(".", n=1).str[-1].str.upper()),
        }
    )

    result: dict[str, str] = {}

    for ticker in tickers:
        matches = lookup.loc[
            lookup["ticker"] == ticker,
            "provider_symbol",
        ]

        if matches.empty:
            raise ValueError(f"Ticker {ticker!r} was not found by AKShare.")

        if len(matches) > 1:
            symbols = ", ".join(matches.tolist())
            raise ValueError(
                f"Ticker {ticker!r} maps to multiple AKShare symbols: {symbols}"
            )

        result[ticker] = matches.iloc[0]

    return result


class MarketDataConnectionError(RuntimeError):
    """Raised when the configured market-data source is unreachable."""


def download_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str | None,
    max_attempts: int = 3,
) -> pd.Series:
    """Download one adjusted closing-price series from Sina via AKShare."""

    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            data = ak.stock_us_daily(
                symbol=ticker,
                adjust="qfq",
            )

            if data.empty:
                raise RuntimeError(
                    f"AKShare returned no data for {ticker}."
                )

            required_columns = {"date", "close"}
            missing_columns = required_columns - set(data.columns)

            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise RuntimeError(
                    f"AKShare data for {ticker} is missing "
                    f"columns: {missing}"
                )

            dates = pd.to_datetime(
                data["date"],
                errors="coerce",
            )

            prices = pd.to_numeric(
                data["close"],
                errors="coerce",
            )

            series = pd.Series(
                data=prices.to_numpy(),
                index=dates,
                name=ticker,
                dtype="float64",
            )

            series = series.loc[series.index.notna()]
            series = series.loc[
                ~series.index.duplicated(keep="last")
            ]
            series = series.sort_index()
            series = series.dropna()

            start = pd.Timestamp(start_date)
            series = series.loc[series.index >= start]

            if end_date is not None:
                end = pd.Timestamp(end_date)
                series = series.loc[series.index <= end]

            if series.empty:
                raise RuntimeError(
                    f"No valid prices remained for {ticker} "
                    "after applying the configured date range."
                )
        
            return series

        except (
            ProxyError,
            Timeout,
            RequestsConnectionError,
        ) as error:
            last_error = error

            if attempt < max_attempts - 1:
                time.sleep(2**attempt)

    raise MarketDataConnectionError(
        f"Unable to download market data for {ticker} "
        f"after {max_attempts} attempts."
    ) from last_error

def download_prices(
    config: BacktestConfig,
) -> pd.DataFrame:
    """Download adjusted daily closing prices through AKShare."""

    series_list: list[pd.Series] = []

    for ticker in config.tickers:
        series = download_single_ticker(
            ticker=ticker,
            start_date=config.start_date,
            end_date=config.end_date,
        )

        series_list.append(series)

    prices = pd.concat(
        series_list,
        axis="columns",
    )

    return clean_prices(
        prices=prices,
        expected_tickers=config.tickers,
    )


def clean_prices(
    prices: pd.DataFrame,
    expected_tickers: tuple[str, ...],
) -> pd.DataFrame:
    """Normalize and validate a closing-price matrix."""

    cleaned = prices.copy()

    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.sort_index()
    cleaned = cleaned.loc[~cleaned.index.duplicated(keep="last")]

    cleaned.columns = [str(column).strip().upper() for column in cleaned.columns]

    missing_tickers = set(expected_tickers) - set(cleaned.columns)

    if missing_tickers:
        missing = ", ".join(sorted(missing_tickers))
        raise ValueError(f"Missing downloaded tickers: {missing}")

    cleaned = cleaned.loc[:, list(expected_tickers)]

    # First remove dates on which every asset is missing.
    cleaned = cleaned.dropna(how="all")

    # Baseline project uses a common complete-data period.
    cleaned = cleaned.dropna(how="any")

    validate_prices(
        prices=cleaned,
        expected_tickers=expected_tickers,
    )

    return cleaned


def validate_prices(
    prices: pd.DataFrame,
    expected_tickers: tuple[str, ...],
) -> None:
    """Validate the processed closing-price matrix."""

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

    non_numeric_columns = [
        column
        for column in prices.columns
        if not pd.api.types.is_numeric_dtype(prices[column])
    ]

    if non_numeric_columns:
        raise TypeError(f"Non-numeric price columns: {non_numeric_columns}")

    if (prices <= 0).any().any():
        invalid_columns = prices.columns[
            (prices <= 0).any()
        ].tolist()

        raise ValueError(
            "Ratio-based return calculations require strictly "
            "positive adjusted prices. Non-positive values were "
            f"found in: {invalid_columns}. Consider using backward-"
            "adjusted prices or a later start date."
        )


def get_price_cache_path(
    config: BacktestConfig,
) -> Path:
    """Return the processed-price cache path."""

    ticker_key = "_".join(config.tickers)

    return config.processed_data_dir / f"prices_akshare_{ticker_key}.parquet"


def save_prices(
    prices: pd.DataFrame,
    path: Path,
) -> None:
    """Save processed prices as Parquet."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prices.to_parquet(path)


def read_cached_prices(
    path: Path,
) -> pd.DataFrame:
    """Read processed prices from a Parquet cache."""

    prices = pd.read_parquet(path)
    prices.index = pd.to_datetime(prices.index)

    return prices


def load_prices(
    config: BacktestConfig,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load cached prices or download them through AKShare."""

    cache_path = get_price_cache_path(config)

    if cache_path.exists() and not refresh:
        prices = read_cached_prices(cache_path)

        validate_prices(
            prices=prices,
            expected_tickers=config.tickers,
        )

        return prices

    prices = download_prices(config)

    save_prices(
        prices=prices,
        path=cache_path,
    )

    return prices
