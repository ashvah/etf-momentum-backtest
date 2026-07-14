"""Market-data loading and processing package."""

from etf_momentum_backtest.data.pipeline import (
    load_prices,
)
from etf_momentum_backtest.data.provider import (
    MarketDataDownloadError,
)


__all__ = [
    "MarketDataDownloadError",
    "load_prices",
]
