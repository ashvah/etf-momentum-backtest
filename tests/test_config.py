from pathlib import Path

import pytest

from etf_momentum_backtest.config import (
    DEFAULT_TICKERS,
    BacktestConfig,
)


def test_default_config_is_valid() -> None:
    config = BacktestConfig()

    assert config.tickers == DEFAULT_TICKERS
    assert config.lookback_days == 126
    assert config.top_k == 2
    assert config.transaction_cost_rate == 0.001
    assert config.initial_capital == 1_000_000.0


def test_custom_tickers_are_supported() -> None:
    config = BacktestConfig(
        tickers=("SPY", "IWM", "EFA"),
        top_k=2,
    )

    assert config.tickers == (
        "SPY",
        "IWM",
        "EFA",
    )


def test_tickers_are_normalized() -> None:
    config = BacktestConfig(
        tickers=(
            " spy ",
            "qqq",
            "Gld",
        ),
        top_k=2,
    )

    assert config.tickers == (
        "SPY",
        "QQQ",
        "GLD",
    )


def test_custom_data_directories_are_supported(
    tmp_path: Path,
) -> None:
    config = BacktestConfig(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
    )

    assert config.raw_data_dir == tmp_path / "raw"
    assert config.processed_data_dir == tmp_path / "processed"


def test_empty_universe_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        BacktestConfig(
            tickers=(),
            top_k=1,
        )


@pytest.mark.parametrize(
    "invalid_ticker",
    [
        "",
        " ",
        "   ",
        "\t",
    ],
)
def test_empty_ticker_symbol_is_rejected(
    invalid_ticker: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty strings",
    ):
        BacktestConfig(
            tickers=(
                "SPY",
                invalid_ticker,
                "GLD",
            ),
            top_k=2,
        )


def test_duplicate_tickers_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        BacktestConfig(
            tickers=(
                "SPY",
                "QQQ",
                "spy",
            ),
            top_k=2,
        )


@pytest.mark.parametrize(
    "lookback_days",
    [
        0,
        -1,
        -126,
    ],
)
def test_non_positive_lookback_is_rejected(
    lookback_days: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="lookback_days must be positive",
    ):
        BacktestConfig(
            lookback_days=lookback_days,
        )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_non_positive_top_k_is_rejected(
    top_k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        BacktestConfig(top_k=top_k)


def test_top_k_cannot_exceed_universe_size() -> None:
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        BacktestConfig(
            tickers=("SPY", "QQQ"),
            top_k=3,
        )


@pytest.mark.parametrize(
    "cost_rate",
    [
        -0.001,
        1.0,
        2.0,
    ],
)
def test_invalid_transaction_cost_is_rejected(
    cost_rate: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="transaction_cost_rate",
    ):
        BacktestConfig(
            transaction_cost_rate=cost_rate,
        )


@pytest.mark.parametrize(
    "initial_capital",
    [
        0.0,
        -1.0,
    ],
)
def test_non_positive_initial_capital_is_rejected(
    initial_capital: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="initial_capital must be positive",
    ):
        BacktestConfig(
            initial_capital=initial_capital,
        )


def test_end_date_must_follow_start_date() -> None:
    with pytest.raises(
        ValueError,
        match="later than start_date",
    ):
        BacktestConfig(
            start_date="2025-01-01",
            end_date="2024-01-01",
        )


def test_equal_start_and_end_dates_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="later than start_date",
    ):
        BacktestConfig(
            start_date="2025-01-01",
            end_date="2025-01-01",
        )


def test_invalid_date_format_is_rejected() -> None:
    with pytest.raises(ValueError):
        BacktestConfig(
            start_date="01/01/2025",
        )
