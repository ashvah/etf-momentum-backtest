import pandas as pd
import pytest

from etf_momentum_backtest.cli import main
from etf_momentum_backtest.config import BacktestConfig


@pytest.fixture(autouse=True)
def mock_load_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent CLI tests from making real network requests."""

    def fake_load_prices(
        config: BacktestConfig,
        refresh: bool = False,
    ) -> pd.DataFrame:
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


def test_cli_uses_default_tickers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main([])

    output = capsys.readouterr().out

    assert "SPY, QQQ, TLT, IEF, GLD" in output
    assert "Top K: 2" in output


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
