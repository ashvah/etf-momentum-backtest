from types import SimpleNamespace

import pandas as pd
import pytest

import etf_momentum_backtest.experiments as experiments
from etf_momentum_backtest.config import BacktestConfig
from etf_momentum_backtest.experiments import (
    ParameterGrid,
    build_parameter_pivot,
    rank_experiments,
    run_parameter_grid,
    select_robust_candidates,
)


def make_config() -> BacktestConfig:
    return BacktestConfig(
        tickers=("SPY", "QQQ", "GLD"),
        start_date="2020-01-01",
        end_date="2024-12-31",
        lookback_days=126,
        top_k=2,
        transaction_cost_rate=0.001,
        initial_capital=10_000.0,
    )


def make_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0],
            "QQQ": [100.0, 102.0, 104.0],
            "GLD": [100.0, 100.5, 101.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-30",
                "2024-01-31",
                "2024-02-01",
            ]
        ),
    )


def test_parameter_grid_sorts_values() -> None:
    grid = ParameterGrid(
        lookback_days=(252, 63, 126),
        top_k=(3, 1, 2),
        transaction_cost_bps=(25, 0, 10),
    )

    assert grid.lookback_days == (
        63,
        126,
        252,
    )
    assert grid.top_k == (1, 2, 3)
    assert grid.transaction_cost_bps == (
        0.0,
        10.0,
        25.0,
    )


def test_parameter_grid_skips_invalid_top_k() -> None:
    grid = ParameterGrid(
        lookback_days=(63,),
        top_k=(1, 2, 4),
        transaction_cost_bps=(0,),
    )

    assert grid.combinations(
        universe_size=3
    ) == [
        (63, 1, 0.0),
        (63, 2, 0.0),
    ]


def test_run_parameter_grid_uses_each_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_configs: list[
        BacktestConfig
    ] = []

    def fake_generate_target_weights(
        prices: pd.DataFrame,
        config: BacktestConfig,
    ) -> pd.DataFrame:
        observed_configs.append(config)

        return pd.DataFrame(
            {
                ticker: [
                    1.0 if position == 0 else 0.0
                ]
                for position, ticker
                in enumerate(config.tickers)
            },
            index=pd.to_datetime(
                ["2024-01-31"]
            ),
        )

    def fake_run_backtest(
        prices: pd.DataFrame,
        target_weights: pd.DataFrame,
        config: BacktestConfig,
    ) -> object:
        return SimpleNamespace(config=config)

    def fake_summarize_performance(
        result: object,
    ) -> object:
        config = result.config

        return SimpleNamespace(
            number_of_rebalances=1,
            final_value=10_000.0
            + config.lookback_days,
            total_return=0.1,
            cagr=config.lookback_days / 10_000,
            annualized_volatility=0.2,
            sharpe_ratio=float(config.top_k),
            sortino_ratio=1.5,
            max_drawdown=-0.1,
            calmar_ratio=0.5,
            annualized_turnover=2.0,
            total_transaction_costs=(
                config.transaction_cost_rate
                * 10_000
            ),
        )

    monkeypatch.setattr(
        experiments,
        "generate_target_weights",
        fake_generate_target_weights,
    )
    monkeypatch.setattr(
        experiments,
        "run_backtest",
        fake_run_backtest,
    )
    monkeypatch.setattr(
        experiments,
        "summarize_performance",
        fake_summarize_performance,
    )

    results = run_parameter_grid(
        prices=make_prices(),
        base_config=make_config(),
        grid=ParameterGrid(
            lookback_days=(63, 126),
            top_k=(1, 2),
            transaction_cost_bps=(0, 10),
        ),
    )

    assert len(results) == 8
    assert len(observed_configs) == 8
    assert set(results["status"]) == {"ok"}

    assert {
        config.lookback_days
        for config in observed_configs
    } == {63, 126}

    assert {
        config.top_k
        for config in observed_configs
    } == {1, 2}

    assert {
        config.transaction_cost_rate
        for config in observed_configs
    } == {0.0, 0.001}


def test_run_parameter_grid_records_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_generate_target_weights(
        prices: pd.DataFrame,
        config: BacktestConfig,
    ) -> pd.DataFrame:
        raise ValueError("bad parameters")

    monkeypatch.setattr(
        experiments,
        "generate_target_weights",
        fake_generate_target_weights,
    )

    results = run_parameter_grid(
        prices=make_prices(),
        base_config=make_config(),
        grid=ParameterGrid(
            lookback_days=(63,),
            top_k=(1,),
            transaction_cost_bps=(0,),
        ),
    )

    assert results.loc[0, "status"] == "error"
    assert results.loc[0, "error"] == (
        "bad parameters"
    )


def test_rank_experiments_orders_metric() -> None:
    results = pd.DataFrame(
        {
            "experiment_id": ["a", "b", "c"],
            "status": ["ok", "error", "ok"],
            "sharpe_ratio": [0.5, 9.0, 1.2],
        }
    )

    ranked = rank_experiments(results)

    assert ranked["experiment_id"].tolist() == [
        "c",
        "a",
    ]
    assert ranked["rank"].tolist() == [1, 2]


def test_build_parameter_pivot() -> None:
    results = pd.DataFrame(
        {
            "status": ["ok", "ok", "ok", "ok"],
            "lookback_days": [63, 63, 126, 126],
            "top_k": [1, 2, 1, 2],
            "transaction_cost_bps": [
                10.0,
                10.0,
                10.0,
                10.0,
            ],
            "sharpe_ratio": [0.7, 0.8, 0.9, 1.0],
        }
    )

    pivot = build_parameter_pivot(
        results,
        metric="sharpe_ratio",
        transaction_cost_bps=10.0,
    )

    assert pivot.loc[63, 1] == pytest.approx(0.7)
    assert pivot.loc[126, 2] == pytest.approx(1.0)


def test_select_robust_candidates() -> None:
    results = pd.DataFrame(
        {
            "status": ["ok", "ok", "error"],
            "sharpe_ratio": [0.8, 1.1, 9.0],
            "max_drawdown": [-0.25, -0.35, -0.01],
            "annualized_turnover": [2.0, 1.0, 0.0],
        }
    )

    selected = select_robust_candidates(
        results,
        min_sharpe=0.7,
        max_drawdown_limit=-0.30,
        max_annualized_turnover=3.0,
    )

    assert len(selected) == 1
    assert selected.iloc[0]["sharpe_ratio"] == (
        pytest.approx(0.8)
    )
