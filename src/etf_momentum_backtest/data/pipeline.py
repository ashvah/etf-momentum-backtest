import pandas as pd

from etf_momentum_backtest.config import BacktestConfig
from etf_momentum_backtest.data.processing import (
    build_price_matrix,
    validate_price_matrix,
)
from etf_momentum_backtest.data.provider import (
    download_raw_ticker,
)
from etf_momentum_backtest.data.repository import (
    get_processed_cache_path,
    get_raw_cache_path,
    read_processed_prices,
    read_raw_data,
    save_processed_prices,
    save_raw_data,
)


def load_prices(
    config: BacktestConfig,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load a processed price matrix.

    When refresh is False:
    1. Use the processed cache if it exists.
    2. Otherwise use cached raw data where available.
    3. Download only missing raw ticker data.
    4. Build and cache the processed matrix.

    When refresh is True:
    1. Download fresh raw data for every ticker.
    2. Replace the raw caches.
    3. Rebuild and replace the processed cache.
    """

    processed_path = get_processed_cache_path(config)

    if processed_path.exists() and not refresh:
        prices = read_processed_prices(processed_path)

        validate_price_matrix(
            prices=prices,
            expected_tickers=config.tickers,
        )

        return prices

    raw_frames: dict[str, pd.DataFrame] = {}

    for ticker in config.tickers:
        raw_path = get_raw_cache_path(
            config=config,
            ticker=ticker,
        )

        if raw_path.exists() and not refresh:
            raw_data = read_raw_data(raw_path)
        else:
            raw_data = download_raw_ticker(
                ticker=ticker,
            )

            save_raw_data(
                raw_data=raw_data,
                path=raw_path,
            )

        raw_frames[ticker] = raw_data

    prices = build_price_matrix(
        raw_data=raw_frames,
        config=config,
    )

    save_processed_prices(
        prices=prices,
        path=processed_path,
    )

    return prices
