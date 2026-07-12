from dataclasses import dataclass
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TICKERS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "TLT",
    "IEF",
    "GLD",
)


@dataclass(frozen=True)
class BacktestConfig:
    """
    Configuration for the backtest.
    """

    tickers: tuple[str, ...] = DEFAULT_TICKERS

    start_date: str = "2005-01-01"
    end_date: str | None = None  # If None, use the latest available date

    lookback_days: int = (
        126  # Number of trading days to look back for momentum calculation
    )
    top_k: int = 2

    transaction_cost_rate: float = 0.001  # 0.1% per trade
    initial_capital: float = 1_000_000.0

    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures"
    results_dir: Path = PROJECT_ROOT / "outputs" / "results"

    def __post_init__(self) -> None:
        """Normalize and validate configuration values."""

        normalized_tickers = tuple(ticker.strip().upper() for ticker in self.tickers)

        if not normalized_tickers:
            raise ValueError("Ticker universe cannot be empty.")

        if any(not ticker for ticker in normalized_tickers):
            raise ValueError("Ticker symbols cannot be empty strings.")

        if len(set(normalized_tickers)) != len(normalized_tickers):
            raise ValueError("Ticker universe contains duplicates.")

        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive.")

        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")

        if self.top_k > len(normalized_tickers):
            raise ValueError("top_k cannot exceed the number of tickers.")

        if not 0 <= self.transaction_cost_rate < 1:
            raise ValueError("transaction_cost_rate must be between 0 and 1.")

        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")

        start = date.fromisoformat(self.start_date)

        if self.end_date is not None:
            end = date.fromisoformat(self.end_date)

            if end <= start:
                raise ValueError("end_date must be later than start_date.")

        object.__setattr__(
            self,
            "tickers",
            normalized_tickers,
        )
