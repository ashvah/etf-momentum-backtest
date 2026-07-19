import argparse
from collections.abc import Sequence

import numpy as np
import pandas as pd

from etf_momentum_backtest.backtest import (
    run_backtest,
)
from etf_momentum_backtest.config import (
    DEFAULT_TICKERS,
    BacktestConfig,
)
from etf_momentum_backtest.data import load_prices
from etf_momentum_backtest.strategy import (
    generate_target_weights,
)

from etf_momentum_backtest.metrics import (
    PerformanceMetrics,
    summarize_performance,
)

from etf_momentum_backtest.benchmarks import (
    run_benchmarks,
)

from etf_momentum_backtest.reporting import (
    save_analysis_outputs,
)

from etf_momentum_backtest.plots import (
    save_analysis_figures,
)

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="etf-momentum",
        description=("Run a monthly ETF momentum rotation backtest."),
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help=("ETF ticker universe. Defaults to SPY QQQ TLT IEF GLD."),
    )

    parser.add_argument(
        "--start-date",
        default="2005-01-01",
        help="Backtest start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end-date",
        default=None,
        help=(
            "Backtest end date in YYYY-MM-DD format. "
            "Defaults to the latest available date."
        ),
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=126,
        help="Momentum lookback in trading days.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of ETFs selected at each rebalance.",
    )

    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help=("Transaction cost in basis points per unit of traded notional."),
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1_000_000.0,
        help="Initial portfolio capital.",
    )

    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help=("Ignore existing market-data caches and download fresh raw data."),
    )

    return parser


def create_config(
    args: argparse.Namespace,
) -> BacktestConfig:
    """Convert parsed command-line arguments into a config."""

    tickers = DEFAULT_TICKERS if args.tickers is None else tuple(args.tickers)

    return BacktestConfig(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        top_k=args.top_k,
        transaction_cost_rate=(args.cost_bps / 10_000),
        initial_capital=args.initial_capital,
    )


def print_configuration(
    config: BacktestConfig,
) -> None:
    """Print the resolved backtest configuration."""

    print("ETF Momentum Rotation Backtest")
    print("-" * 40)

    print(f"Tickers: {', '.join(config.tickers)}")

    print(f"Date range: {config.start_date} to {config.end_date or 'latest'}")

    print(f"Lookback days: {config.lookback_days}")

    print(f"Top K: {config.top_k}")

    print(f"Transaction cost: {config.transaction_cost_rate:.2%}")

    print(f"Initial capital: ${config.initial_capital:,.2f}")


def format_ratio(value: float) -> str:
    """Format a ratio while handling undefined values."""

    if np.isnan(value):
        return "N/A"

    return f"{value:.2f}"


def print_backtest_summary(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    metrics: PerformanceMetrics,
) -> None:
    """Print a performance summary."""

    print()
    print("Data")
    print("-" * 40)

    print(
        f"Loaded {len(prices):,} daily observations "
        f"from {metrics.start_date.date()} "
        f"to {metrics.end_date.date()}."
    )

    print(f"Generated {len(target_weights):,} monthly signals.")

    print(f"Executed {metrics.number_of_rebalances:,} rebalances.")

    print()
    print("Return and Risk")
    print("-" * 40)

    print(f"Final portfolio value: ${metrics.final_value:,.2f}")

    print(f"Total return: {metrics.total_return:.2%}")

    print(f"CAGR: {metrics.cagr:.2%}")

    print(f"Annualized volatility: {metrics.annualized_volatility:.2%}")

    print(f"Sharpe ratio: {format_ratio(metrics.sharpe_ratio)}")

    print(f"Sortino ratio: {format_ratio(metrics.sortino_ratio)}")

    print(f"Maximum drawdown: {metrics.max_drawdown:.2%}")

    print(
        "Maximum drawdown duration: "
        f"{metrics.max_drawdown_duration_periods} "
        "trading periods"
    )

    print(f"Calmar ratio: {format_ratio(metrics.calmar_ratio)}")

    print()
    print("Trading")
    print("-" * 40)

    print(f"Total turnover: {metrics.total_turnover:.2f}x")

    print(f"Annualized turnover: {metrics.annualized_turnover:.2f}x")

    print(
        f"Average turnover per rebalance: {metrics.average_turnover_per_rebalance:.2f}x"
    )

    print(f"Total transaction costs: ${metrics.total_transaction_costs:,.2f}")

    print(
        "Transaction costs / initial capital: "
        f"{metrics.transaction_cost_to_initial_capital:.2%}"
    )


def print_benchmark_comparison(
    strategy_metrics: PerformanceMetrics,
    benchmark_metrics: dict[
        str,
        PerformanceMetrics,
    ],
) -> None:
    """Print strategy and benchmark metrics side by side."""

    rows = {
        "Momentum Strategy": strategy_metrics,
        **benchmark_metrics,
    }

    print()
    print("Benchmark Comparison")
    print("-" * 82)

    print(
        f"{'Portfolio':<30}"
        f"{'CAGR':>10}"
        f"{'Volatility':>12}"
        f"{'Sharpe':>10}"
        f"{'Max DD':>10}"
        f"{'Calmar':>10}"
    )

    print("-" * 82)

    for name, metrics in rows.items():
        print(
            f"{name:<30}"
            f"{metrics.cagr:>10.2%}"
            f"{metrics.annualized_volatility:>12.2%}"
            f"{format_ratio(metrics.sharpe_ratio):>10}"
            f"{metrics.max_drawdown:>10.2%}"
            f"{format_ratio(metrics.calmar_ratio):>10}"
        )


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Run the ETF momentum backtest CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = create_config(args)

        print_configuration(config)

        prices = load_prices(
            config=config,
            refresh=args.refresh_data,
        )

        target_weights = generate_target_weights(
            prices=prices,
            config=config,
        )

        result = run_backtest(
            prices=prices,
            target_weights=target_weights,
            config=config,
        )

        result = run_backtest(
            prices=prices,
            target_weights=target_weights,
            config=config,
        )

        metrics = summarize_performance(result)

        benchmark_results = run_benchmarks(
            prices=prices,
            strategy_target_weights=target_weights,
            config=config,
        )

        benchmark_metrics = {
            name: summarize_performance(benchmark_result)
            for name, benchmark_result in benchmark_results.as_dict().items()
        }

        run_dir = save_analysis_outputs(
            config=config,
            strategy_target_weights=target_weights,
            strategy_result=result,
            strategy_metrics=metrics,
            benchmark_results=benchmark_results,
            benchmark_metrics=benchmark_metrics,
        )

    except (
        ValueError,
        TypeError,
        RuntimeError,
    ) as error:
        parser.error(str(error))

    print_backtest_summary(
        prices=prices,
        target_weights=target_weights,
        metrics=metrics,
    )

    print_benchmark_comparison(
        strategy_metrics=metrics,
        benchmark_metrics=benchmark_metrics,
    )

    print()
    print(f"Results saved to: {run_dir}")
    
    figure_dir = (
        config.figures_dir
        / run_dir.name
    )

    figure_paths = save_analysis_figures(
        strategy_result=result,
        benchmark_results=benchmark_results,
        target_weights=target_weights,
        output_dir=figure_dir,
    )

    print(
        f"Figures saved to: {figure_dir}"
    )
