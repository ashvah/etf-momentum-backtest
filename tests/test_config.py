import pytest

from etf_momentum_backtest.config import BacktestConfig


def test_default_config_is_valid() -> None:
    config = BacktestConfig()

    assert config.tickers == (
        "SPY",
        "QQQ",
        "TLT",
        "IEF",
        "GLD",
    )
    assert config.lookback_days == 126
    assert config.top_k == 2
    assert config.initial_capital == 1_000_000.0


def test_custom_tickers_are_supported() -> None:
    config = BacktestConfig(
        tickers=("SPY", "IWM", "EFA"),
        top_k=2,
    )

    assert config.tickers == ("SPY", "IWM", "EFA")
    assert config.top_k == 2


def test_tickers_are_normalized() -> None:
    config = BacktestConfig(
        tickers=(" spy ", "qqq", "Gld"),
        top_k=2,
    )

    assert config.tickers == ("SPY", "QQQ", "GLD")


def test_empty_universe_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        BacktestConfig(
            tickers=(),
            top_k=1,
        )


def test_duplicate_tickers_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        BacktestConfig(
            tickers=("SPY", "QQQ", "SPY"),
            top_k=2,
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
        match="Ticker symbols cannot be empty strings",
    ):
        BacktestConfig(
            tickers=("SPY", invalid_ticker, "GLD"),
            top_k=2,
        )


def test_top_k_cannot_exceed_universe_size() -> None:
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        BacktestConfig(
            tickers=("SPY", "QQQ"),
            top_k=3,
        )


def test_negative_transaction_cost_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="transaction_cost_rate",
    ):
        BacktestConfig(
            transaction_cost_rate=-0.001,
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
