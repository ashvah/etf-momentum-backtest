import argparse
from collections.abc import Sequence

import pandas as pd

from etf_momentum_backtest.backtest import (
    BacktestResult,
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


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="etf-momentum",
        description=(
            "Run a monthly ETF momentum rotation backtest."
        ),
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help=(
            "ETF ticker universe. "
            "Defaults to SPY QQQ TLT IEF GLD."
        ),
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
        help=(
            "Transaction cost in basis points per unit "
            "of traded notional."
        ),
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
        help=(
            "Ignore existing market-data caches and "
            "download fresh raw data."
        ),
    )

    return parser


def create_config(
    args: argparse.Namespace,
) -> BacktestConfig:
    """Convert parsed command-line arguments into a config."""

    tickers = (
        DEFAULT_TICKERS
        if args.tickers is None
        else tuple(args.tickers)
    )

    return BacktestConfig(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        top_k=args.top_k,
        transaction_cost_rate=(
            args.cost_bps / 10_000
        ),
        initial_capital=args.initial_capital,
    )


def print_configuration(
    config: BacktestConfig,
) -> None:
    """Print the resolved backtest configuration."""

    print("ETF Momentum Rotation Backtest")
    print("-" * 40)

    print(
        "Tickers: "
        f"{', '.join(config.tickers)}"
    )

    print(
        f"Date range: {config.start_date} "
        f"to {config.end_date or 'latest'}"
    )

    print(
        f"Lookback days: "
        f"{config.lookback_days}"
    )

    print(
        f"Top K: {config.top_k}"
    )

    print(
        "Transaction cost: "
        f"{config.transaction_cost_rate:.2%}"
    )

    print(
        "Initial capital: "
        f"${config.initial_capital:,.2f}"
    )


def print_backtest_summary(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    result: BacktestResult,
    config: BacktestConfig,
) -> None:
    """Print a basic summary of the completed backtest."""

    first_date = prices.index.min().date()
    last_date = prices.index.max().date()

    final_value = float(
        result.portfolio_value.iloc[-1]
    )

    total_return = (
        final_value
        / config.initial_capital
        - 1.0
    )

    total_turnover = float(
        result.turnover.sum()
    )

    total_transaction_costs = float(
        result.transaction_costs.sum()
    )

    print()
    print("Data")
    print("-" * 40)

    print(
        f"Loaded {len(prices):,} daily observations "
        f"from {first_date} to {last_date}."
    )

    print(
        f"Generated {len(target_weights):,} "
        "monthly signals."
    )

    print(
        f"Executed "
        f"{len(result.execution_targets):,} "
        "rebalances."
    )

    print()
    print("Backtest Results")
    print("-" * 40)

    print(
        "Final portfolio value: "
        f"${final_value:,.2f}"
    )

    print(
        f"Total return: {total_return:.2%}"
    )

    print(
        f"Total turnover: {total_turnover:.2f}x"
    )

    print(
        "Total transaction costs: "
        f"${total_transaction_costs:,.2f}"
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

    except (
        ValueError,
        TypeError,
        RuntimeError,
    ) as error:
        parser.error(str(error))

    print_backtest_summary(
        prices=prices,
        target_weights=target_weights,
        result=result,
        config=config,
    )