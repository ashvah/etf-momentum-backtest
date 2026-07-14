import time
from collections.abc import Callable

import akshare as ak
import pandas as pd
from requests.exceptions import RequestException


DATA_PROVIDER = "akshare_sina"
ADJUSTMENT_METHOD = "qfq"


class MarketDataDownloadError(RuntimeError):
    """Raised when raw market data cannot be downloaded."""


def download_raw_ticker(
    ticker: str,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Download one ticker's raw daily data from AKShare.

    The returned DataFrame is kept as close as possible to the
    provider response. Date filtering, column extraction, and
    strategy-specific validation are performed later.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    normalized_ticker = ticker.strip().upper()
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            data = ak.stock_us_daily(
                symbol=normalized_ticker,
                adjust=ADJUSTMENT_METHOD,
            )
        except RequestException as error:
            last_error = error
        else:
            if isinstance(data, pd.DataFrame) and not data.empty:
                return data.copy()

            last_error = RuntimeError(
                f"AKShare returned no data for {normalized_ticker}."
            )

        if attempt < max_attempts - 1:
            sleep(2**attempt)

    raise MarketDataDownloadError(
        f"Unable to download raw market data for "
        f"{normalized_ticker} after {max_attempts} attempts."
    ) from last_error
