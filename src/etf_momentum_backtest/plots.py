from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from etf_momentum_backtest.backtest import BacktestResult
from etf_momentum_backtest.benchmarks import BenchmarkResults
from matplotlib.ticker import PercentFormatter


def build_portfolio_value_frame(
    strategy_result: BacktestResult,
    benchmark_results: BenchmarkResults,
) -> pd.DataFrame:
    """Combine strategy and benchmark portfolio values."""

    values = {
        "Momentum Strategy": (
            strategy_result.portfolio_value
        ),
        **{
            name: result.portfolio_value
            for name, result
            in benchmark_results.as_dict().items()
        },
    }

    return pd.DataFrame(values)


def plot_portfolio_growth(
    strategy_result: BacktestResult,
    benchmark_results: BenchmarkResults,
    output_path: Path,
) -> None:
    """Plot normalized strategy and benchmark growth."""

    values = build_portfolio_value_frame(
        strategy_result=strategy_result,
        benchmark_results=benchmark_results,
    )

    normalized = values.div(values.iloc[0])

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 6),
    )

    normalized.plot(
        ax=axis,
    )

    axis.set_title(
        "Strategy and Benchmark Growth"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Growth of $1")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
    )
    plt.close(figure)

def calculate_drawdowns(
    portfolio_values: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate drawdowns for several portfolios."""

    high_water_marks = portfolio_values.cummax()

    return (
        portfolio_values
        / high_water_marks
        - 1.0
    )


def plot_drawdowns(
    strategy_result: BacktestResult,
    benchmark_results: BenchmarkResults,
    output_path: Path,
) -> None:
    """Plot strategy and benchmark drawdowns."""

    values = build_portfolio_value_frame(
        strategy_result=strategy_result,
        benchmark_results=benchmark_results,
    )

    drawdowns = calculate_drawdowns(values)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 6),
    )

    drawdowns.plot(
        ax=axis,
    )

    axis.set_title(
        "Strategy and Benchmark Drawdowns"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")
    axis.yaxis.set_major_formatter(
        lambda value, position: f"{value:.0%}"
    )
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
    )
    plt.close(figure)

def plot_target_weights(
    target_weights: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot monthly strategy target weights."""

    if target_weights.empty:
        raise ValueError(
            "target_weights cannot be empty."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 6),
    )

    target_weights.plot.area(
        ax=axis,
    )

    axis.set_title(
        "Monthly Momentum Target Weights"
    )
    axis.set_xlabel("Signal Date")
    axis.set_ylabel("Target Weight")
    axis.set_ylim(0.0, 1.0)
    axis.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

def save_analysis_figures(
    strategy_result: BacktestResult,
    benchmark_results: BenchmarkResults,
    target_weights: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Save all standard analysis figures."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = [
        output_dir / "portfolio_growth.png",
        output_dir / "drawdowns.png",
        output_dir / "target_weights.png",
    ]

    plot_portfolio_growth(
        strategy_result=strategy_result,
        benchmark_results=benchmark_results,
        output_path=paths[0],
    )

    plot_drawdowns(
        strategy_result=strategy_result,
        benchmark_results=benchmark_results,
        output_path=paths[1],
    )

    plot_target_weights(
        target_weights=target_weights,
        output_path=paths[2],
    )

    return paths