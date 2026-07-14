import pandas as pd
import pytest

import etf_momentum_backtest.cli as cli
from etf_momentum_backtest.backtest import BacktestResult
from etf_momentum_backtest.config import BacktestConfig


@pytest.fixture(autouse=True)
def mock_backtest_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[object]]:
    """Replace external data, strategy, and backtest operations."""

    calls: dict[str, list[object]] = {
        "refresh": [],
        "configs": [],
        "prices": [],
        "target_weights": [],
    }

    def fake_load_prices(
        config: BacktestConfig,
        refresh: bool = False,
    ) -> pd.DataFrame:
        calls["refresh"].append(refresh)
        calls["configs"].append(config)

        prices = pd.DataFrame(
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
            dtype="float64",
        )

        calls["prices"].append(prices)

        return prices

    def fake_generate_target_weights(
        prices: pd.DataFrame,
        config: BacktestConfig,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(
            0.0,
            index=pd.to_datetime(["2024-01-31"]),
            columns=config.tickers,
            dtype="float64",
        )

        selected_tickers = list(config.tickers[: config.top_k])

        weights.loc[
            pd.Timestamp("2024-01-31"),
            selected_tickers,
        ] = 1.0 / config.top_k

        calls["target_weights"].append(weights)

        return weights

    def fake_run_backtest(
        prices: pd.DataFrame,
        target_weights: pd.DataFrame,
        config: BacktestConfig,
    ) -> BacktestResult:
        index = prices.index

        portfolio_value = pd.Series(
            [
                config.initial_capital,
                config.initial_capital,
                config.initial_capital * 1.05,
            ],
            index=index,
            name="portfolio_value",
            dtype="float64",
        )

        daily_returns = portfolio_value.pct_change(fill_method=None).fillna(0.0)
        daily_returns.name = "daily_return"

        actual_weights = pd.DataFrame(
            0.0,
            index=index,
            columns=config.tickers,
            dtype="float64",
        )

        selected_tickers = list(config.tickers[: config.top_k])

        actual_weights.loc[
            index[-1],
            selected_tickers,
        ] = 1.0 / config.top_k

        cash_balance = pd.Series(
            [
                config.initial_capital,
                config.initial_capital,
                0.0,
            ],
            index=index,
            name="cash_balance",
            dtype="float64",
        )

        turnover = pd.Series(
            [
                0.0,
                0.0,
                1.0,
            ],
            index=index,
            name="turnover",
            dtype="float64",
        )

        transaction_costs = pd.Series(
            [
                0.0,
                0.0,
                100.0,
            ],
            index=index,
            name="transaction_cost",
            dtype="float64",
        )

        execution_targets = target_weights.copy()
        execution_targets.index = pd.DatetimeIndex(
            ["2024-02-01"],
            name="execution_date",
        )

        return BacktestResult(
            portfolio_value=portfolio_value,
            daily_returns=daily_returns,
            actual_weights=actual_weights,
            cash_balance=cash_balance,
            turnover=turnover,
            transaction_costs=transaction_costs,
            execution_targets=execution_targets,
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
    assert "Lookback days: 126" in output
    assert "Top K: 2" in output
    assert "Transaction cost: 0.10%" in output
    assert "Initial capital: $1,000,000.00" in output

    assert "Loaded 3 daily observations" in output
    assert "Generated 1 monthly signals" in output
    assert "Executed 1 rebalances" in output

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
    assert "Generated 1 monthly signals" in output
    assert "Executed 1 rebalances" in output


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

    assert mock_backtest_pipeline["refresh"] == [True]


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

    assert mock_backtest_pipeline["refresh"] == [False]


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

    config = mock_backtest_pipeline["configs"][0]

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

    assert config.transaction_cost_rate == pytest.approx(0.0005)

    assert config.initial_capital == pytest.approx(500_000.0)


def test_cli_passes_config_to_strategy(
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
            "GLD",
            "--top-k",
            "2",
        ]
    )

    target_weights = mock_backtest_pipeline["target_weights"][0]

    assert isinstance(
        target_weights,
        pd.DataFrame,
    )

    assert tuple(target_weights.columns) == (
        "SPY",
        "QQQ",
        "GLD",
    )

    assert target_weights.loc[
        "2024-01-31",
        "SPY",
    ] == pytest.approx(0.5)

    assert target_weights.loc[
        "2024-01-31",
        "QQQ",
    ] == pytest.approx(0.5)

    assert target_weights.loc[
        "2024-01-31",
        "GLD",
    ] == pytest.approx(0.0)


def test_cli_rejects_top_k_larger_than_universe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
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


def test_cli_rejects_invalid_date_range(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--start-date",
                "2025-01-01",
                "--end-date",
                "2024-01-01",
            ]
        )

    error_output = capsys.readouterr().err

    assert "end_date must be later" in error_output


def test_cli_rejects_negative_transaction_cost(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--cost-bps",
                "-1",
            ]
        )

    error_output = capsys.readouterr().err

    assert "transaction_cost_rate" in error_output
