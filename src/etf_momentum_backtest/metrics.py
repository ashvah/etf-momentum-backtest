from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from etf_momentum_backtest.backtest import BacktestResult


DEFAULT_PERIODS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class DrawdownStatistics:
    """Summary of portfolio drawdowns."""

    drawdown: pd.Series
    max_drawdown: float
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    max_drawdown_duration_periods: int


@dataclass(frozen=True)
class PerformanceMetrics:
    """Summary statistics for a completed backtest."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    observations: int

    initial_value: float
    final_value: float
    total_return: float
    cagr: float

    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float

    max_drawdown: float
    max_drawdown_peak_date: pd.Timestamp
    max_drawdown_trough_date: pd.Timestamp
    max_drawdown_recovery_date: pd.Timestamp | None
    max_drawdown_duration_periods: int
    calmar_ratio: float

    total_turnover: float
    annualized_turnover: float
    average_turnover_per_rebalance: float
    number_of_rebalances: int

    total_transaction_costs: float
    transaction_cost_to_initial_capital: float


def validate_portfolio_value(
    portfolio_value: pd.Series,
) -> None:
    """Validate a portfolio-value series."""

    if not isinstance(portfolio_value, pd.Series):
        raise TypeError("portfolio_value must be a pandas Series.")

    if portfolio_value.empty:
        raise ValueError("portfolio_value cannot be empty.")

    if len(portfolio_value) < 2:
        raise ValueError("portfolio_value must contain at least two observations.")

    if not isinstance(
        portfolio_value.index,
        pd.DatetimeIndex,
    ):
        raise TypeError("portfolio_value index must be a DatetimeIndex.")

    if not portfolio_value.index.is_monotonic_increasing:
        raise ValueError("portfolio_value index must be sorted.")

    if portfolio_value.index.has_duplicates:
        raise ValueError("portfolio_value index contains duplicate dates.")

    if not pd.api.types.is_numeric_dtype(portfolio_value):
        raise TypeError("portfolio_value must contain numeric values.")

    if portfolio_value.isna().any():
        raise ValueError("portfolio_value contains missing values.")

    if not np.isfinite(portfolio_value.to_numpy(dtype="float64")).all():
        raise ValueError("portfolio_value contains infinite values.")

    if (portfolio_value <= 0).any():
        raise ValueError("portfolio_value must be strictly positive.")


def validate_daily_returns(
    daily_returns: pd.Series,
) -> None:
    """Validate a daily-return series."""

    if not isinstance(daily_returns, pd.Series):
        raise TypeError("daily_returns must be a pandas Series.")

    if daily_returns.empty:
        raise ValueError("daily_returns cannot be empty.")

    if not isinstance(
        daily_returns.index,
        pd.DatetimeIndex,
    ):
        raise TypeError("daily_returns index must be a DatetimeIndex.")

    if not daily_returns.index.is_monotonic_increasing:
        raise ValueError("daily_returns index must be sorted.")

    if daily_returns.index.has_duplicates:
        raise ValueError("daily_returns index contains duplicate dates.")

    if not pd.api.types.is_numeric_dtype(daily_returns):
        raise TypeError("daily_returns must contain numeric values.")

    if daily_returns.isna().any():
        raise ValueError("daily_returns contains missing values.")

    if not np.isfinite(daily_returns.to_numpy(dtype="float64")).all():
        raise ValueError("daily_returns contains infinite values.")

    if (daily_returns <= -1.0).any():
        raise ValueError("daily_returns cannot be less than or equal to -100%.")


def calculate_elapsed_years(
    index: pd.DatetimeIndex,
) -> float:
    """Calculate elapsed calendar years between first and last date."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex.")

    if len(index) < 2:
        raise ValueError("At least two dates are required.")

    elapsed_days = (index[-1] - index[0]).total_seconds() / 86_400

    if elapsed_days <= 0:
        raise ValueError("The final date must be later than the initial date.")

    return elapsed_days / DAYS_PER_YEAR


def calculate_total_return(
    portfolio_value: pd.Series,
) -> float:
    """Calculate cumulative portfolio return."""

    validate_portfolio_value(portfolio_value)

    return float(portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1.0)


def calculate_cagr(
    portfolio_value: pd.Series,
) -> float:
    """Calculate compound annual growth rate."""

    validate_portfolio_value(portfolio_value)

    elapsed_years = calculate_elapsed_years(portfolio_value.index)

    growth_multiple = float(portfolio_value.iloc[-1] / portfolio_value.iloc[0])

    return growth_multiple ** (1.0 / elapsed_years) - 1.0


def calculate_annualized_volatility(
    daily_returns: pd.Series,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Calculate annualized sample volatility."""

    validate_daily_returns(daily_returns)

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    if len(daily_returns) < 2:
        return float("nan")

    daily_volatility = float(daily_returns.std(ddof=1))

    return daily_volatility * sqrt(periods_per_year)


def convert_annual_rate_to_periodic(
    annual_rate: float,
    periods_per_year: int,
) -> float:
    """Convert an effective annual rate into a periodic rate."""

    if annual_rate <= -1.0:
        raise ValueError("annual_rate must be greater than -100%.")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


def calculate_sharpe_ratio(
    daily_returns: pd.Series,
    annual_risk_free_rate: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Calculate the annualized Sharpe ratio."""

    validate_daily_returns(daily_returns)

    periodic_risk_free_rate = convert_annual_rate_to_periodic(
        annual_rate=annual_risk_free_rate,
        periods_per_year=periods_per_year,
    )

    excess_returns = daily_returns - periodic_risk_free_rate

    if len(excess_returns) < 2:
        return float("nan")

    excess_volatility = float(excess_returns.std(ddof=1))

    if np.isclose(excess_volatility, 0.0):
        return float("nan")

    return float(excess_returns.mean() / excess_volatility * sqrt(periods_per_year))


def calculate_sortino_ratio(
    daily_returns: pd.Series,
    annual_risk_free_rate: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Calculate the annualized Sortino ratio.

    Downside deviation is calculated from returns below the
    periodic risk-free return.
    """

    validate_daily_returns(daily_returns)

    periodic_risk_free_rate = convert_annual_rate_to_periodic(
        annual_rate=annual_risk_free_rate,
        periods_per_year=periods_per_year,
    )

    excess_returns = daily_returns - periodic_risk_free_rate

    downside_returns = np.minimum(
        excess_returns.to_numpy(dtype="float64"),
        0.0,
    )

    periodic_downside_deviation = float(np.sqrt(np.mean(np.square(downside_returns))))

    if np.isclose(
        periodic_downside_deviation,
        0.0,
    ):
        return float("nan")

    annualized_excess_return = float(excess_returns.mean() * periods_per_year)

    annualized_downside_deviation = periodic_downside_deviation * sqrt(periods_per_year)

    return annualized_excess_return / annualized_downside_deviation


def calculate_drawdown_series(
    portfolio_value: pd.Series,
) -> pd.Series:
    """Calculate drawdown relative to the previous high-water mark."""

    validate_portfolio_value(portfolio_value)

    high_water_mark = portfolio_value.cummax()

    drawdown = portfolio_value / high_water_mark - 1.0

    drawdown.name = "drawdown"

    return drawdown


def calculate_max_drawdown_duration(
    drawdown: pd.Series,
) -> int:
    """Calculate the longest number of consecutive underwater periods."""

    if not isinstance(drawdown, pd.Series):
        raise TypeError("drawdown must be a pandas Series.")

    if drawdown.empty:
        raise ValueError("drawdown cannot be empty.")

    if drawdown.isna().any():
        raise ValueError("drawdown contains missing values.")

    longest_duration = 0
    current_duration = 0

    for value in drawdown:
        if value < 0:
            current_duration += 1
            longest_duration = max(
                longest_duration,
                current_duration,
            )
        else:
            current_duration = 0

    return longest_duration


def analyze_drawdowns(
    portfolio_value: pd.Series,
) -> DrawdownStatistics:
    """Calculate drawdown magnitude, dates, and duration."""

    drawdown = calculate_drawdown_series(portfolio_value)

    trough_date = pd.Timestamp(drawdown.idxmin())

    max_drawdown = float(drawdown.loc[trough_date])

    if np.isclose(max_drawdown, 0.0):
        peak_date = pd.Timestamp(portfolio_value.index[0])

        recovery_date: pd.Timestamp | None = peak_date
    else:
        values_before_trough = portfolio_value.loc[:trough_date]

        peak_date = pd.Timestamp(values_before_trough.idxmax())

        peak_value = float(portfolio_value.loc[peak_date])

        values_after_trough = portfolio_value.loc[trough_date:]

        recovery_candidates = values_after_trough[values_after_trough >= peak_value]

        if recovery_candidates.empty:
            recovery_date = None
        else:
            recovery_date = pd.Timestamp(recovery_candidates.index[0])

    max_duration = calculate_max_drawdown_duration(drawdown)

    return DrawdownStatistics(
        drawdown=drawdown,
        max_drawdown=max_drawdown,
        peak_date=peak_date,
        trough_date=trough_date,
        recovery_date=recovery_date,
        max_drawdown_duration_periods=max_duration,
    )


def calculate_calmar_ratio(
    cagr: float,
    max_drawdown: float,
) -> float:
    """Calculate CAGR divided by absolute maximum drawdown."""

    drawdown_magnitude = abs(max_drawdown)

    if np.isclose(drawdown_magnitude, 0.0):
        return float("nan")

    return cagr / drawdown_magnitude


def validate_result_consistency(
    result: BacktestResult,
) -> None:
    """Validate alignment and internal consistency of backtest outputs."""

    validate_portfolio_value(result.portfolio_value)

    validate_daily_returns(result.daily_returns)

    reference_index = result.portfolio_value.index

    aligned_series = {
        "daily_returns": result.daily_returns,
        "cash_balance": result.cash_balance,
        "turnover": result.turnover,
        "transaction_costs": (result.transaction_costs),
    }

    for name, series in aligned_series.items():
        if not series.index.equals(reference_index):
            raise ValueError(f"{name} index does not match portfolio_value index.")

    if not result.actual_weights.index.equals(reference_index):
        raise ValueError("actual_weights index does not match portfolio_value index.")

    if (result.turnover < 0).any():
        raise ValueError("turnover contains negative values.")

    if (result.transaction_costs < 0).any():
        raise ValueError("transaction_costs contains negative values.")

    calculated_returns = result.portfolio_value.pct_change(fill_method=None).iloc[1:]

    recorded_returns = result.daily_returns.iloc[1:]

    if not np.allclose(
        calculated_returns.to_numpy(dtype="float64"),
        recorded_returns.to_numpy(dtype="float64"),
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError("daily_returns are inconsistent with portfolio_value.")


def summarize_performance(
    result: BacktestResult,
    annual_risk_free_rate: float = 0.0,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> PerformanceMetrics:
    """Calculate a complete performance summary."""

    validate_result_consistency(result)

    portfolio_value = result.portfolio_value

    # run_backtest sets the first return to zero because no
    # preceding portfolio value exists. Exclude that artificial
    # observation from distribution-based statistics.
    realized_returns = result.daily_returns.iloc[1:]

    total_return = calculate_total_return(portfolio_value)

    cagr = calculate_cagr(portfolio_value)

    annualized_volatility = calculate_annualized_volatility(
        daily_returns=realized_returns,
        periods_per_year=periods_per_year,
    )

    sharpe_ratio = calculate_sharpe_ratio(
        daily_returns=realized_returns,
        annual_risk_free_rate=(annual_risk_free_rate),
        periods_per_year=periods_per_year,
    )

    sortino_ratio = calculate_sortino_ratio(
        daily_returns=realized_returns,
        annual_risk_free_rate=(annual_risk_free_rate),
        periods_per_year=periods_per_year,
    )

    drawdown_statistics = analyze_drawdowns(portfolio_value)

    calmar_ratio = calculate_calmar_ratio(
        cagr=cagr,
        max_drawdown=(drawdown_statistics.max_drawdown),
    )

    elapsed_years = calculate_elapsed_years(portfolio_value.index)

    total_turnover = float(result.turnover.sum())

    annualized_turnover = total_turnover / elapsed_years

    number_of_rebalances = len(result.execution_targets)

    if number_of_rebalances == 0:
        average_turnover_per_rebalance = float("nan")
    else:
        average_turnover_per_rebalance = total_turnover / number_of_rebalances

    total_transaction_costs = float(result.transaction_costs.sum())

    initial_value = float(portfolio_value.iloc[0])

    return PerformanceMetrics(
        start_date=pd.Timestamp(portfolio_value.index[0]),
        end_date=pd.Timestamp(portfolio_value.index[-1]),
        observations=len(portfolio_value),
        initial_value=initial_value,
        final_value=float(portfolio_value.iloc[-1]),
        total_return=total_return,
        cagr=cagr,
        annualized_volatility=(annualized_volatility),
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=(drawdown_statistics.max_drawdown),
        max_drawdown_peak_date=(drawdown_statistics.peak_date),
        max_drawdown_trough_date=(drawdown_statistics.trough_date),
        max_drawdown_recovery_date=(drawdown_statistics.recovery_date),
        max_drawdown_duration_periods=(
            drawdown_statistics.max_drawdown_duration_periods
        ),
        calmar_ratio=calmar_ratio,
        total_turnover=total_turnover,
        annualized_turnover=(annualized_turnover),
        average_turnover_per_rebalance=(average_turnover_per_rebalance),
        number_of_rebalances=(number_of_rebalances),
        total_transaction_costs=(total_transaction_costs),
        transaction_cost_to_initial_capital=(total_transaction_costs / initial_value),
    )
