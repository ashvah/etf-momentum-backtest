from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import pandas as pd

from etf_momentum_backtest.backtest import run_backtest
from etf_momentum_backtest.config import BacktestConfig
from etf_momentum_backtest.metrics import summarize_performance
from etf_momentum_backtest.strategy import generate_target_weights


@dataclass(frozen=True)
class ParameterGrid:
    """Parameter values used in a momentum-strategy sweep."""

    lookback_days: tuple[int, ...] = (63, 126, 189, 252)
    top_k: tuple[int, ...] = (1, 2, 3)
    transaction_cost_bps: tuple[float, ...] = (0.0, 10.0, 25.0)

    def __post_init__(self) -> None:
        lookbacks = tuple(int(value) for value in self.lookback_days)
        top_ks = tuple(int(value) for value in self.top_k)
        costs = tuple(float(value) for value in self.transaction_cost_bps)

        if not lookbacks:
            raise ValueError("lookback_days cannot be empty.")

        if not top_ks:
            raise ValueError("top_k cannot be empty.")

        if not costs:
            raise ValueError("transaction_cost_bps cannot be empty.")

        if any(value <= 0 for value in lookbacks):
            raise ValueError(
                "All lookback_days values must be positive."
            )

        if any(value <= 0 for value in top_ks):
            raise ValueError(
                "All top_k values must be positive."
            )

        if any(value < 0 for value in costs):
            raise ValueError(
                "transaction_cost_bps cannot be negative."
            )

        if len(set(lookbacks)) != len(lookbacks):
            raise ValueError(
                "lookback_days contains duplicate values."
            )

        if len(set(top_ks)) != len(top_ks):
            raise ValueError(
                "top_k contains duplicate values."
            )

        if len(set(costs)) != len(costs):
            raise ValueError(
                "transaction_cost_bps contains duplicate values."
            )

        object.__setattr__(
            self,
            "lookback_days",
            tuple(sorted(lookbacks)),
        )
        object.__setattr__(
            self,
            "top_k",
            tuple(sorted(top_ks)),
        )
        object.__setattr__(
            self,
            "transaction_cost_bps",
            tuple(sorted(costs)),
        )

    def combinations(
        self,
        universe_size: int,
    ) -> list[tuple[int, int, float]]:
        """Return valid grid combinations.

        top_k values larger than the asset universe are omitted.
        """

        if universe_size <= 0:
            raise ValueError(
                "universe_size must be positive."
            )

        return [
            (lookback, top_k, cost_bps)
            for lookback, top_k, cost_bps in product(
                self.lookback_days,
                self.top_k,
                self.transaction_cost_bps,
            )
            if top_k <= universe_size
        ]


def build_experiment_id(
    lookback_days: int,
    top_k: int,
    transaction_cost_bps: float,
) -> str:
    """Build a deterministic ID for one parameter combination."""

    cost_text = format(
        transaction_cost_bps,
        "g",
    ).replace(".", "p")

    return (
        f"lb{lookback_days}_"
        f"top{top_k}_"
        f"cost{cost_text}bps"
    )


def run_parameter_grid(
    prices: pd.DataFrame,
    base_config: BacktestConfig,
    grid: ParameterGrid,
    *,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Run all valid combinations using one shared price matrix.

    The function retains only summary metrics, so it does not keep every
    full BacktestResult in memory.
    """

    combinations = grid.combinations(
        universe_size=len(base_config.tickers),
    )

    if not combinations:
        raise ValueError(
            "The parameter grid contains no valid combinations."
        )

    rows: list[dict[str, object]] = []

    for lookback_days, top_k, cost_bps in combinations:
        experiment_id = build_experiment_id(
            lookback_days=lookback_days,
            top_k=top_k,
            transaction_cost_bps=cost_bps,
        )

        config = replace(
            base_config,
            lookback_days=lookback_days,
            top_k=top_k,
            transaction_cost_rate=cost_bps / 10_000,
        )

        common_row: dict[str, object] = {
            "experiment_id": experiment_id,
            "lookback_days": lookback_days,
            "top_k": top_k,
            "transaction_cost_bps": cost_bps,
        }

        try:
            target_weights = generate_target_weights(
                prices=prices,
                config=config,
            )

            result = run_backtest(
                prices=prices,
                target_weights=target_weights,
                config=config,
            )

            metrics = summarize_performance(result)

            rows.append(
                {
                    **common_row,
                    "status": "ok",
                    "error": None,
                    "signals": len(target_weights),
                    "rebalances": metrics.number_of_rebalances,
                    "final_value": metrics.final_value,
                    "total_return": metrics.total_return,
                    "cagr": metrics.cagr,
                    "annualized_volatility": (
                        metrics.annualized_volatility
                    ),
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "sortino_ratio": metrics.sortino_ratio,
                    "max_drawdown": metrics.max_drawdown,
                    "calmar_ratio": metrics.calmar_ratio,
                    "annualized_turnover": (
                        metrics.annualized_turnover
                    ),
                    "total_transaction_costs": (
                        metrics.total_transaction_costs
                    ),
                }
            )

        except (
            ValueError,
            TypeError,
            RuntimeError,
            ArithmeticError,
        ) as error:
            if not continue_on_error:
                raise

            rows.append(
                {
                    **common_row,
                    "status": "error",
                    "error": str(error),
                    "signals": pd.NA,
                    "rebalances": pd.NA,
                    "final_value": float("nan"),
                    "total_return": float("nan"),
                    "cagr": float("nan"),
                    "annualized_volatility": float("nan"),
                    "sharpe_ratio": float("nan"),
                    "sortino_ratio": float("nan"),
                    "max_drawdown": float("nan"),
                    "calmar_ratio": float("nan"),
                    "annualized_turnover": float("nan"),
                    "total_transaction_costs": float("nan"),
                }
            )

    results = pd.DataFrame(rows)

    return results.sort_values(
        [
            "lookback_days",
            "top_k",
            "transaction_cost_bps",
        ],
        kind="stable",
    ).reset_index(drop=True)


def rank_experiments(
    results: pd.DataFrame,
    *,
    metric: str = "sharpe_ratio",
    ascending: bool = False,
) -> pd.DataFrame:
    """Rank successful experiments by one result column."""

    required_columns = {
        "experiment_id",
        "status",
        metric,
    }

    missing_columns = required_columns.difference(
        results.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing result columns: "
            f"{sorted(missing_columns)}"
        )

    ranked = results.loc[
        results["status"].eq("ok")
    ].dropna(
        subset=[metric]
    ).copy()

    if ranked.empty:
        raise ValueError(
            "No successful experiments can be ranked."
        )

    ranked = ranked.sort_values(
        metric,
        ascending=ascending,
        kind="stable",
    ).reset_index(drop=True)

    ranked.insert(
        0,
        "rank",
        range(1, len(ranked) + 1),
    )

    return ranked


def build_parameter_pivot(
    results: pd.DataFrame,
    *,
    metric: str,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Create a lookback-by-top_k sensitivity table."""

    required_columns = {
        "status",
        "lookback_days",
        "top_k",
        "transaction_cost_bps",
        metric,
    }

    missing_columns = required_columns.difference(
        results.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing result columns: "
            f"{sorted(missing_columns)}"
        )

    selected = results.loc[
        results["status"].eq("ok")
        & results["transaction_cost_bps"].eq(
            float(transaction_cost_bps)
        )
    ]

    if selected.empty:
        raise ValueError(
            "No successful experiments match "
            f"transaction_cost_bps={transaction_cost_bps:g}."
        )

    pivot = selected.pivot(
        index="lookback_days",
        columns="top_k",
        values=metric,
    )

    pivot.index.name = "lookback_days"
    pivot.columns.name = "top_k"

    return pivot.sort_index().sort_index(
        axis="columns"
    )


def select_robust_candidates(
    results: pd.DataFrame,
    *,
    min_sharpe: float | None = None,
    max_drawdown_limit: float | None = None,
    max_annualized_turnover: float | None = None,
) -> pd.DataFrame:
    """Filter successful runs using explicit robustness constraints.

    max_drawdown_limit should be negative. For example, -0.30 keeps
    experiments whose maximum drawdown is no worse than -30%.
    """

    required_columns = {
        "status",
        "sharpe_ratio",
        "max_drawdown",
        "annualized_turnover",
    }

    missing_columns = required_columns.difference(
        results.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing result columns: "
            f"{sorted(missing_columns)}"
        )

    selected = results.loc[
        results["status"].eq("ok")
    ].copy()

    if min_sharpe is not None:
        selected = selected.loc[
            selected["sharpe_ratio"] >= min_sharpe
        ]

    if max_drawdown_limit is not None:
        if max_drawdown_limit > 0:
            raise ValueError(
                "max_drawdown_limit must be zero or negative."
            )

        selected = selected.loc[
            selected["max_drawdown"]
            >= max_drawdown_limit
        ]

    if max_annualized_turnover is not None:
        if max_annualized_turnover < 0:
            raise ValueError(
                "max_annualized_turnover cannot be negative."
            )

        selected = selected.loc[
            selected["annualized_turnover"]
            <= max_annualized_turnover
        ]

    return selected.reset_index(drop=True)


def save_experiment_results(
    results: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save grid results to CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    return output_path
