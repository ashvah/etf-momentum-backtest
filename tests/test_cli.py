import pandas as pd
import pytest

from etf_momentum_backtest.cli import main
from etf_momentum_backtest.config import (
    BacktestConfig,
)


@pytest.fixture(autouse=True)
def mock_load_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> list[bool]:
    """Prevent CLI tests from accessing the network."""

    received_refresh_values: list[bool] = []

    def fake_load_prices(
        config: BacktestConfig,
        refresh: bool = False,
    ) -> pd.DataFrame:
        received_refresh_values.append(refresh)

        return pd.DataFrame(
            {ticker: [100.0, 101.0] for ticker in config.tickers},
            index=pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
        )

    monkeypatch.setattr(
        "etf_momentum_backtest.cli.load_prices",
        fake_load_prices,
    )

    return received_refresh_values


def test_cli_uses_default_tickers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main([])

    output = capsys.readouterr().out

    assert "SPY, QQQ, TLT, IEF, GLD" in output
    assert "Top K: 2" in output
    assert "Loaded 2 daily observations" in output


def test_cli_accepts_custom_tickers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--tickers",
            "spy",
            "iwm",
            "gld",
            "--top-k",
            "2",
            "--cost-bps",
            "5",
        ]
    )

    output = capsys.readouterr().out

    assert "SPY, IWM, GLD" in output
    assert "Top K: 2" in output
    assert "Transaction cost: 0.05%" in output
    assert "Loaded 2 daily observations" in output


def test_cli_passes_refresh_flag(
    mock_load_prices: list[bool],
) -> None:
    main(
        [
            "--tickers",
            "SPY",
            "--top-k",
            "1",
            "--refresh-data",
        ]
    )

    assert mock_load_prices == [True]


def test_cli_does_not_refresh_by_default(
    mock_load_prices: list[bool],
) -> None:
    main(
        [
            "--tickers",
            "SPY",
            "--top-k",
            "1",
        ]
    )

    assert mock_load_prices == [False]


def test_cli_rejects_top_k_larger_than_universe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--tickers",
                "SPY",
                "QQQ",
                "--top-k",
                "3",
            ]
        )

    error_output = capsys.readouterr().err

    assert "top_k cannot exceed" in error_output
