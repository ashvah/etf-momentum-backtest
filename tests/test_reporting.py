import json
from pathlib import Path

import pandas as pd

from etf_momentum_backtest.backtest import (
    BacktestResult,
)
from etf_momentum_backtest.benchmarks import (
    BenchmarkResults,
)
from etf_momentum_backtest.config import (
    BacktestConfig,
)
from etf_momentum_backtest.metrics import (
    summarize_performance,
)
from etf_momentum_backtest.reporting import (
    build_run_name,
    save_analysis_outputs,
)


def make_config(
    tmp_path: Path,
) -> BacktestConfig:
    return BacktestConfig(
        tickers=("SPY", "QQQ"),
        start_date="2024-01-01",
        end_date="2024-12-31",
        lookback_days=63,
        top_k=1,
        transaction_cost_rate=0.0005,
        initial_capital=10_000.0,
        results_dir=tmp_path / "results",
    )


def make_result() -> BacktestResult:
    index = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
        ]
    )

    portfolio_value = pd.Series(
        [
            10_000.0,
            10_000.0,
            10_500.0,
        ],
        index=index,
        name="portfolio_value",
    )

    daily_returns = portfolio_value.pct_change(fill_method=None).fillna(0.0)

    daily_returns.name = "daily_return"

    return BacktestResult(
        portfolio_value=portfolio_value,
        daily_returns=daily_returns,
        actual_weights=pd.DataFrame(
            {
                "SPY": [0.0, 1.0, 1.0],
                "QQQ": [0.0, 0.0, 0.0],
            },
            index=index,
        ),
        cash_balance=pd.Series(
            [10_000.0, 0.0, 0.0],
            index=index,
            name="cash_balance",
        ),
        turnover=pd.Series(
            [0.0, 1.0, 0.0],
            index=index,
            name="turnover",
        ),
        transaction_costs=pd.Series(
            [0.0, 5.0, 0.0],
            index=index,
            name="transaction_cost",
        ),
        execution_targets=pd.DataFrame(
            {
                "SPY": [1.0],
                "QQQ": [0.0],
            },
            index=pd.DatetimeIndex(
                ["2024-01-03"],
                name="execution_date",
            ),
        ),
    )


def test_build_run_name(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    assert build_run_name(config) == ("SPY_QQQ_20240101_20241231_lb63_top1_cost5bps")


def test_analysis_outputs_are_saved(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    strategy_result = make_result()
    strategy_metrics = summarize_performance(strategy_result)

    benchmark_results = BenchmarkResults(
        benchmark_ticker="SPY",
        market_buy_and_hold=make_result(),
        equal_weight_buy_and_hold=make_result(),
        monthly_equal_weight=make_result(),
    )

    benchmark_metrics = {
        name: summarize_performance(result)
        for name, result in benchmark_results.as_dict().items()
    }

    target_weights = pd.DataFrame(
        {
            "SPY": [1.0],
            "QQQ": [0.0],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )

    run_dir = save_analysis_outputs(
        config=config,
        strategy_target_weights=target_weights,
        strategy_result=strategy_result,
        strategy_metrics=strategy_metrics,
        benchmark_results=benchmark_results,
        benchmark_metrics=benchmark_metrics,
    )

    assert run_dir.exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "target_weights.csv").exists()

    assert (run_dir / "strategy" / "portfolio_value.csv").exists()

    assert (run_dir / "strategy" / "metrics.json").exists()

    assert (run_dir / "performance_comparison.csv").exists()

    assert (run_dir / "benchmarks" / "SPY_Buy_and_Hold" / "metrics.json").exists()


def test_saved_config_is_valid_json(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    result = make_result()
    metrics = summarize_performance(result)

    benchmark_results = BenchmarkResults(
        benchmark_ticker="SPY",
        market_buy_and_hold=result,
        equal_weight_buy_and_hold=result,
        monthly_equal_weight=result,
    )

    benchmark_metrics = {name: metrics for name in benchmark_results.as_dict()}

    target_weights = pd.DataFrame(
        {
            "SPY": [1.0],
            "QQQ": [0.0],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )

    run_dir = save_analysis_outputs(
        config=config,
        strategy_target_weights=target_weights,
        strategy_result=result,
        strategy_metrics=metrics,
        benchmark_results=benchmark_results,
        benchmark_metrics=benchmark_metrics,
    )

    with (run_dir / "config.json").open(
        encoding="utf-8",
    ) as file:
        saved_config = json.load(file)

    assert saved_config["tickers"] == [
        "SPY",
        "QQQ",
    ]

    assert saved_config["lookback_days"] == 63
    assert saved_config["top_k"] == 1
