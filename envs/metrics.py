"""Financial performance and risk-adjusted metrics calculation module."""

import numpy as np


def calculate_cumulative_return(portfolio_values: np.ndarray) -> float:
    """Calculates total cumulative return in percentage."""
    values = np.asarray(portfolio_values, dtype=np.float64)
    if len(values) < 2 or values[0] <= 0:
        return 0.0
    return float((values[-1] - values[0]) / values[0] * 100.0)


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Calculates Maximum Drawdown (MDD) in percentage."""
    values = np.asarray(portfolio_values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    peaks = np.maximum.accumulate(values)
    drawdowns = (values - peaks) / (peaks + 1e-8)
    max_dd = float(np.min(drawdowns))
    return float(abs(max_dd) * 100.0)


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculates annualized Sharpe Ratio."""
    rets = np.asarray(returns, dtype=np.float64)
    if len(rets) < 2:
        return 0.0
    excess_returns = rets - (risk_free_rate / periods_per_year)
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)
    if std_excess < 1e-8:
        return 0.0
    return float(np.sqrt(periods_per_year) * (mean_excess / std_excess))


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculates annualized Sortino Ratio."""
    rets = np.asarray(returns, dtype=np.float64)
    if len(rets) < 2:
        return 0.0
    excess_returns = rets - (risk_free_rate / periods_per_year)
    mean_excess = np.mean(excess_returns)
    downside_returns = np.minimum(excess_returns, 0.0)
    downside_std = np.sqrt(np.mean(downside_returns**2))
    if downside_std < 1e-8:
        return float(np.sqrt(periods_per_year) * mean_excess) if mean_excess > 0 else 0.0
    return float(np.sqrt(periods_per_year) * (mean_excess / downside_std))


def calculate_performance_metrics(
    portfolio_values: np.ndarray,
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Computes a dictionary of standard portfolio performance metrics."""
    return {
        "cumulative_return_pct": calculate_cumulative_return(portfolio_values),
        "max_drawdown_pct": calculate_max_drawdown(portfolio_values),
        "annualized_sharpe": calculate_sharpe_ratio(returns, risk_free_rate=risk_free_rate),
        "annualized_sortino": calculate_sortino_ratio(returns, risk_free_rate=risk_free_rate),
    }
