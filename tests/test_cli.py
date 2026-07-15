import pandas as pd
import pytest

import etf_momentum_backtest.cli as cli
from etf_momentum_backtest.backtest import BacktestResult
from etf_momentum_backtest.benchmarks import BenchmarkResults
from etf_momentum_backtest.config import BacktestConfig
from pathlib import Path


def make_backtest_result(
    prices: pd.DataFrame,
    config: BacktestConfig,
    final_growth: float,
    transaction_cost: float = 100.0,
) -> BacktestResult:
    """Create an internally consistent fake backtest result."""

    index = prices.index

    portfolio_value = pd.Series(
        [
            config.initial_capital,
            config.initial_capital,
            config.initial_capital * final_growth,
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
        [0.0, 0.0, 1.0],
        index=index,
        name="turnover",
        dtype="float64",
    )

    transaction_costs = pd.Series(
        [0.0, 0.0, transaction_cost],
        index=index,
        name="transaction_cost",
        dtype="float64",
    )

    execution_targets = pd.DataFrame(
        0.0,
        index=pd.DatetimeIndex(
            [index[-1]],
            name="execution_date",
        ),
        columns=config.tickers,
        dtype="float64",
    )

    execution_targets.loc[
        index[-1],
        selected_tickers,
    ] = 1.0 / config.top_k

    return BacktestResult(
        portfolio_value=portfolio_value,
        daily_returns=daily_returns,
        actual_weights=actual_weights,
        cash_balance=cash_balance,
        turnover=turnover,
        transaction_costs=transaction_costs,
        execution_targets=execution_targets,
    )


@pytest.fixture(autouse=True)
def mock_backtest_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[object]]:
    """Replace data, strategy, backtest, and benchmark operations."""

    calls: dict[str, list[object]] = {
        "refresh": [],
        "configs": [],
        "target_weights": [],
        "benchmark_tickers": [],
        "saved_outputs": [],
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
            dtype="float64",
        )

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
        return make_backtest_result(
            prices=prices,
            config=config,
            final_growth=1.05,
        )

    def fake_run_benchmarks(
        prices: pd.DataFrame,
        strategy_target_weights: pd.DataFrame,
        config: BacktestConfig,
        benchmark_ticker: str = "SPY",
    ) -> BenchmarkResults:
        calls["benchmark_tickers"].append(benchmark_ticker)

        return BenchmarkResults(
            benchmark_ticker=benchmark_ticker,
            market_buy_and_hold=make_backtest_result(
                prices=prices,
                config=config,
                final_growth=1.04,
                transaction_cost=90.0,
            ),
            equal_weight_buy_and_hold=(
                make_backtest_result(
                    prices=prices,
                    config=config,
                    final_growth=1.03,
                    transaction_cost=80.0,
                )
            ),
            monthly_equal_weight=(
                make_backtest_result(
                    prices=prices,
                    config=config,
                    final_growth=1.02,
                    transaction_cost=120.0,
                )
            ),
        )

    def fake_save_analysis_outputs(
        config: BacktestConfig,
        strategy_target_weights: pd.DataFrame,
        strategy_result: BacktestResult,
        strategy_metrics: object,
        benchmark_results: BenchmarkResults,
        benchmark_metrics: dict[str, object],
    ) -> Path:
        calls["saved_outputs"].append(
            {
                "config": config,
                "strategy_target_weights": (strategy_target_weights),
                "strategy_result": strategy_result,
                "benchmark_results": (benchmark_results),
                "benchmark_metrics": (benchmark_metrics),
            }
        )

        return Path("/tmp/fake-results")

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

    monkeypatch.setattr(
        cli,
        "run_benchmarks",
        fake_run_benchmarks,
    )

    monkeypatch.setattr(
        cli,
        "save_analysis_outputs",
        fake_save_analysis_outputs,
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


def test_cli_prints_benchmark_comparison(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main([])

    output = capsys.readouterr().out

    assert "Benchmark Comparison" in output
    assert "Momentum Strategy" in output
    assert "SPY Buy & Hold" in output
    assert "Equal Weight Buy & Hold" in output
    assert "Monthly Equal Weight" in output

    assert "CAGR" in output
    assert "Volatility" in output
    assert "Sharpe" in output
    assert "Max DD" in output
    assert "Calmar" in output


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


def test_cli_uses_spy_as_default_benchmark(
    mock_backtest_pipeline: dict[
        str,
        list[object],
    ],
) -> None:
    cli.main([])

    assert mock_backtest_pipeline["benchmark_tickers"] == ["SPY"]


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


def test_cli_saves_analysis_outputs(
    mock_backtest_pipeline: dict[
        str,
        list[object],
    ],
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main([])

    output = capsys.readouterr().out

    assert len(mock_backtest_pipeline["saved_outputs"]) == 1

    assert "Results saved to:" in output
    assert "fake-results" in output
