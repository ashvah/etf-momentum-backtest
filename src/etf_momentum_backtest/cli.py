import argparse
from collections.abc import Sequence

from etf_momentum_backtest.config import (
    DEFAULT_TICKERS,
    BacktestConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="etf-momentum",
        description=("Run a monthly ETF momentum rotation backtest."),
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="TICKER",
        help=("ETF ticker symbols. Example: --tickers SPY QQQ TLT GLD"),
    )

    parser.add_argument(
        "--start-date",
        default="2005-01-01",
        help="Backtest start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional backtest end date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=126,
        help="Momentum lookback period in trading days.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of highest-momentum assets to hold.",
    )

    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="One-way transaction cost in basis points.",
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1_000_000.0,
        help="Initial portfolio capital.",
    )

    return parser


def create_config(
    args: argparse.Namespace,
) -> BacktestConfig:
    tickers = tuple(args.tickers) if args.tickers is not None else DEFAULT_TICKERS

    return BacktestConfig(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        top_k=args.top_k,
        transaction_cost_rate=args.cost_bps / 10_000,
        initial_capital=args.initial_capital,
    )


def main(
    argv: Sequence[str] | None = None,
) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = create_config(args)
    except ValueError as error:
        parser.error(str(error))

    print("ETF Momentum Rotation Backtest")
    print(f"Tickers: {', '.join(config.tickers)}")
    print(f"Start date: {config.start_date}")
    print(f"End date: {config.end_date or 'latest available'}")
    print(f"Lookback: {config.lookback_days} trading days")
    print(f"Top K: {config.top_k}")
    print(f"Transaction cost: {config.transaction_cost_rate:.2%}")
    print(f"Initial capital: {config.initial_capital:,.2f}")


if __name__ == "__main__":
    main()
