from collections.abc import Mapping

import numpy as np
import pandas as pd

from etf_momentum_backtest.config import BacktestConfig


REQUIRED_RAW_COLUMNS = frozenset(
    {
        "date",
        "close",
    }
)


def validate_raw_ticker_data(
    raw_data: pd.DataFrame,
    ticker: str,
) -> None:
    """Validate the minimum structure of one raw ticker dataset."""

    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError(f"Raw data for {ticker} must be a pandas DataFrame.")

    if raw_data.empty:
        raise ValueError(f"Raw data for {ticker} is empty.")

    missing_columns = REQUIRED_RAW_COLUMNS - set(raw_data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))

        raise ValueError(f"Raw data for {ticker} is missing columns: {missing}")


def extract_close_series(
    raw_data: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str | None,
) -> pd.Series:
    """Convert one raw ticker dataset into a closing-price series."""

    validate_raw_ticker_data(
        raw_data=raw_data,
        ticker=ticker,
    )

    dates = pd.to_datetime(
        raw_data["date"],
        errors="coerce",
    )

    close_prices = pd.to_numeric(
        raw_data["close"],
        errors="coerce",
    )

    series = pd.Series(
        data=close_prices.to_numpy(),
        index=dates,
        name=ticker,
        dtype="float64",
    )

    # Remove records whose dates could not be parsed.
    series = series.loc[series.index.notna()]

    # Remove values that could not be parsed as numbers.
    series = series.dropna()

    # If the provider returned multiple records for one date,
    # keep the final observation.
    series = series.loc[~series.index.duplicated(keep="last")]

    series = series.sort_index()

    start = pd.Timestamp(start_date)
    series = series.loc[series.index >= start]

    if end_date is not None:
        end = pd.Timestamp(end_date)
        series = series.loc[series.index <= end]

    if series.empty:
        raise ValueError(
            f"No valid prices remained for {ticker} after "
            "applying the configured date range."
        )

    return series


def build_price_matrix(
    raw_data: Mapping[str, pd.DataFrame],
    config: BacktestConfig,
) -> pd.DataFrame:
    """Build a common adjusted closing-price matrix."""

    missing_raw_tickers = set(config.tickers) - set(raw_data)

    if missing_raw_tickers:
        missing = ", ".join(sorted(missing_raw_tickers))

        raise ValueError(f"Missing raw data for tickers: {missing}")

    series_list = [
        extract_close_series(
            raw_data=raw_data[ticker],
            ticker=ticker,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        for ticker in config.tickers
    ]

    prices = pd.concat(series_list, axis="columns", sort=False)

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    prices = prices.loc[~prices.index.duplicated(keep="last")]

    prices.columns = [str(column).strip().upper() for column in prices.columns]

    prices = prices.loc[
        :,
        list(config.tickers),
    ]

    # Remove dates on which every asset is missing.
    prices = prices.dropna(how="all")

    # The baseline strategy uses a static universe and therefore
    # keeps only dates for which every configured ETF has a price.
    prices = prices.dropna(how="any")

    validate_price_matrix(
        prices=prices,
        expected_tickers=config.tickers,
    )

    return prices


def validate_price_matrix(
    prices: pd.DataFrame,
    expected_tickers: tuple[str, ...],
) -> None:
    """Validate a price matrix used by ratio-based strategies."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("Prices must be a pandas DataFrame.")

    if prices.empty:
        raise ValueError("Price data is empty.")

    if not isinstance(
        prices.index,
        pd.DatetimeIndex,
    ):
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

    if not np.isfinite(prices.to_numpy(dtype="float64")).all():
        raise ValueError("Price data contains infinite values.")

    non_positive_mask = prices <= 0

    if non_positive_mask.any().any():
        invalid_columns = prices.columns[non_positive_mask.any()].tolist()

        raise ValueError(
            "Ratio-based return calculations require strictly "
            "positive adjusted prices. Non-positive values were "
            f"found in: {invalid_columns}. Consider using a later "
            "start date or a different total-return data source."
        )
