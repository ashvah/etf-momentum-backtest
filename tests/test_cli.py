from types import SimpleNamespace

import pandas as pd
import pytest

import etf_momentum_backtest.cli as cli
from etf_momentum_backtest.config import (
    BacktestConfig,
)


@pytest.fixture(autouse=True)
def mock_backtest_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[object]]:
    """Replace the data, strategy, and backtest layers."""

    calls: dict[str, list[object]] = {
        "refresh": [],
        "configs": [],
    }

    def fake_load_prices(
        config: BacktestConfig,
        refresh: bool = False,
    ) -> pd.DataFrame:
        calls["refresh"].append(refresh)
        calls["configs"].append(config)

        return pd.DataFrame(
            {
                ticker: [
                    100.0,
                    101.0,
                    102.0,
                ]
                for ticker in config.tickers
            },
            index=pd.to_datetime(
                [
                    "2024-01-30",
                    "2024-01-31",
                    "2024-02-01",
                ]
            ),
        )

    def fake_generate_target_weights(
        prices: pd.DataFrame,
        config: BacktestConfig,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(
            0.0,
            index=pd.to_datetime(
                ["2024-01-31"]
            ),
            columns=config.tickers,
        )

        selected = list(
            config.tickers[: config.top_k]
        )

        weights.loc[
            pd.Timestamp("2024-01-31"),
            selected,
        ] = 1.0 / config.top_k

        return weights

    def fake_run_backtest(
        prices: pd.DataFrame,
        target_weights: pd.DataFrame,
        config: BacktestConfig,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            portfolio_value=pd.Series(
                [
                    config.initial_capital,
                    config.initial_capital,
                    config.initial_capital
                    * 1.05,
                ],
                index=prices.index,
                name="portfolio_value",
            ),
            turnover=pd.Series(
                [0.0, 0.0, 1.0],
                index=prices.index,
                name="turnover",
            ),
            transaction_costs=pd.Series(
                [0.0, 0.0, 100.0],
                index=prices.index,
                name="transaction_costs",
            ),
            execution_targets=target_weights.copy(),
        )

    monkeypatch.setattr(
        cli,
        "load_prices",
        fake_load_prices,
    )

    monkeypatch.setattr(
        cli,
        "generate_target_weights",
        fake_generate_target_weights,
    )

    monkeypatch.setattr(
        cli,
        "run_backtest",
        fake_run_backtest,
    )

    return calls


def test_cli_uses_default_tickers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main([])

    output = capsys.readouterr().out

    assert "SPY, QQQ, TLT, IEF, GLD" in output
    assert "Top K: 2" in output
    assert "Loaded 3 daily observations" in output
    assert "Generated 1 monthly signals" in output
    assert "Final portfolio value: $1,050,000.00" in output
    assert "Total return: 5.00%" in output


def test_cli_accepts_custom_tickers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main(
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


def test_cli_passes_refresh_flag(
    mock_backtest_pipeline: dict[
        str,
        list[object],
    ],
) -> None:
    cli.main(
        [
            "--tickers",
            "SPY",
            "--top-k",
            "1",
            "--refresh-data",
        ]
    )

    assert mock_backtest_pipeline[
        "refresh"
    ] == [True]


def test_cli_does_not_refresh_by_default(
    mock_backtest_pipeline: dict[
        str,
        list[object],
    ],
) -> None:
    cli.main(
        [
            "--tickers",
            "SPY",
            "--top-k",
            "1",
        ]
    )

    assert mock_backtest_pipeline[
        "refresh"
    ] == [False]


def test_cli_passes_custom_config(
    mock_backtest_pipeline: dict[
        str,
        list[object],
    ],
) -> None:
    cli.main(
        [
            "--tickers",
            "SPY",
            "QQQ",
            "--start-date",
            "2015-01-01",
            "--end-date",
            "2020-12-31",
            "--lookback-days",
            "63",
            "--top-k",
            "1",
            "--cost-bps",
            "5",
            "--initial-capital",
            "500000",
        ]
    )

    config = mock_backtest_pipeline[
        "configs"
    ][0]

    assert isinstance(
        config,
        BacktestConfig,
    )

    assert config.tickers == (
        "SPY",
        "QQQ",
    )

    assert config.start_date == "2015-01-01"
    assert config.end_date == "2020-12-31"
    assert config.lookback_days == 63
    assert config.top_k == 1

    assert (
        config.transaction_cost_rate
        == pytest.approx(0.0005)
    )

    assert config.initial_capital == 500_000.0


def test_cli_rejects_invalid_top_k(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(
        SystemExit,
    ):
        cli.main(
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