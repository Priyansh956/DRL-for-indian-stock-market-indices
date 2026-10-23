import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from envs.india_trading_env import IndiaMarketTradingEnv


def test_environment_initialization():
    mock_features = pd.DataFrame(np.random.randn(100, 10))
    mock_prices = pd.DataFrame(100 + np.cumsum(np.random.randn(100, 3) * 0.1, axis=0))

    env = IndiaMarketTradingEnv(mock_features, mock_prices)
    obs, info = env.reset()

    # Check observation shape: features + portfolio weights
    assert obs.shape == (10 + 3,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert np.isclose(np.sum(env.weights), 1.0)
    assert env.portfolio_value == 100000.0
    assert "portfolio_value" in info
    assert "step" in info
    assert info["step"] == 0


def test_action_step_weights():
    mock_features = pd.DataFrame(np.random.randn(100, 10))
    mock_prices = pd.DataFrame(100 + np.cumsum(np.random.randn(100, 3) * 0.1, axis=0))

    env = IndiaMarketTradingEnv(mock_features, mock_prices)
    env.reset()

    # Pass an arbitrary action vector
    action = np.array([0.2, 0.5, 0.3])
    obs, reward, terminated, truncated, info = env.step(action)

    # Verify weights are clipped/normalized and sum to 1
    assert np.isclose(np.sum(env.weights), 1.0)
    assert np.all(env.weights >= 0.0)
    assert not terminated
    assert not truncated
    assert env.observation_space.contains(obs)


def test_action_normalization_edge_cases():
    mock_features = pd.DataFrame(np.random.randn(50, 4))
    mock_prices = pd.DataFrame(np.ones((50, 3)) * 100.0)
    env = IndiaMarketTradingEnv(mock_features, mock_prices)
    env.reset()

    # Case 1: All zeros action should default to uniform weights
    _, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(env.weights, np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32))

    # Case 2: Negative values should be clipped to 0 and remaining normalized
    _, _, _, _, _ = env.step(np.array([-2.0, 0.5, 0.5]))
    assert np.allclose(env.weights, np.array([0.0, 0.5, 0.5], dtype=np.float32))

    # Case 3: Unscaled positive numbers (summing > 1)
    _, _, _, _, _ = env.step(np.array([10.0, 20.0, 70.0]))
    assert np.allclose(env.weights, np.array([0.1, 0.2, 0.7], dtype=np.float32))

    # Case 4: Non-finite values (NaN / Inf) should fall back safely without error
    _, _, _, _, _ = env.step(np.array([np.nan, np.inf, -np.inf]))
    assert np.isclose(np.sum(env.weights), 1.0)
    assert np.all(env.weights >= 0.0)

    # Case 5: Incorrect action shape/length
    _, _, _, _, _ = env.step(np.array([1.0, 2.0]))
    assert np.isclose(np.sum(env.weights), 1.0)


def test_portfolio_value_and_transaction_cost():
    # 3 time steps, 2 assets
    features = pd.DataFrame(np.zeros((3, 2)))
    # Day 0: [100, 100], Day 1: [110, 100], Day 2: [110, 120]
    prices = pd.DataFrame([[100.0, 100.0], [110.0, 100.0], [110.0, 120.0]])

    env = IndiaMarketTradingEnv(
        features,
        prices,
        transaction_cost_pct=0.001,
        initial_capital=100000.0,
    )
    env.reset()
    # At reset: weights = [1.0, 0.0]

    # Step 1: Keep 100% in asset 0 (turnover = 0, asset 0 +10%)
    obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
    expected_value_step1 = 100000.0 * 1.10
    assert np.isclose(env.portfolio_value, expected_value_step1, rtol=1e-5)
    assert np.isclose(info["turnover"], 0.0)
    assert np.isclose(info["transaction_cost"], 0.0)

    # Step 2: Switch 100% to asset 1 (turnover = |0 - 1| + |1 - 0| = 2.0)
    # Asset 1 increases from 100 to 120 (+20%)
    # Transaction cost = 2.0 * 0.001 = 0.002 (0.2%)
    # Net return = 20% - 0.2% = 19.8%
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 1.0]))
    expected_value_step2 = expected_value_step1 * (1.0 + 0.20 - 0.002)
    assert np.isclose(env.portfolio_value, expected_value_step2, rtol=1e-4)
    assert np.isclose(info["turnover"], 2.0)
    assert terminated


def test_risk_adjusted_reward_rolling_sharpe():
    # 30 steps with constant return
    features = pd.DataFrame(np.zeros((30, 2)))
    prices = pd.DataFrame(np.linspace(100.0, 130.0, 30).reshape(-1, 1))

    env = IndiaMarketTradingEnv(features, prices, reward_window=20)
    env.reset()

    # Steps 1 to 19: return immediate portfolio return
    for _ in range(15):
        _, reward, _, _, info = env.step(np.array([1.0]))
        assert np.isfinite(reward)

    # Steps 20+: rolling Sharpe ratio
    for _ in range(10):
        _, reward, _, _, info = env.step(np.array([1.0]))
        assert np.isfinite(reward)


def test_zero_volatility_reward_handling():
    # Constant prices -> zero volatility in returns
    features = pd.DataFrame(np.zeros((30, 2)))
    prices = pd.DataFrame(np.full((30, 1), 100.0))

    env = IndiaMarketTradingEnv(features, prices, reward_window=5)
    env.reset()

    for _ in range(10):
        _, reward, _, _, _ = env.step(np.array([1.0]))
        # Reward must be finite, zero, and not NaN or Inf
        assert np.isfinite(reward)
        assert np.isclose(reward, 0.0)


def test_episode_termination_and_boundary():
    # Small 4-step dataset
    features = pd.DataFrame(np.random.randn(4, 2))
    prices = pd.DataFrame(np.random.uniform(100, 110, (4, 1)))

    env = IndiaMarketTradingEnv(features, prices)
    env.reset()

    # Step 1: current_step = 1
    _, _, term1, _, _ = env.step(np.array([1.0]))
    assert not term1

    # Step 2: current_step = 2
    _, _, term2, _, _ = env.step(np.array([1.0]))
    assert not term2

    # Step 3: current_step = 3 (last index: 4 - 1 = 3)
    obs, reward, term3, trunc, info = env.step(np.array([1.0]))
    assert term3
    assert not trunc
    assert env.current_step == 3

    # Step 4 (after termination): Should return safely without IndexError
    obs_extra, reward_extra, term_extra, _, _ = env.step(np.array([1.0]))
    assert term_extra
    assert reward_extra == 0.0


def test_bankruptcy_termination():
    features = pd.DataFrame(np.zeros((4, 1)))
    # Asset drops to 0 (100% loss)
    prices = pd.DataFrame([[100.0], [0.0], [0.0], [0.0]])

    env = IndiaMarketTradingEnv(features, prices)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(np.array([1.0]))
    assert terminated
    assert env.portfolio_value == 0.0


def test_input_validation():
    # Length mismatch
    with pytest.raises(ValueError, match="Feature rows"):
        IndiaMarketTradingEnv(np.zeros((10, 2)), np.zeros((8, 1)))

    # Less than 2 rows
    with pytest.raises(ValueError, match="at least 2 rows"):
        IndiaMarketTradingEnv(np.zeros((1, 2)), np.zeros((1, 1)))

    # 1D inputs auto-reshaped
    env = IndiaMarketTradingEnv(np.zeros(10), np.zeros(10))
    assert env.n_features == 1
    assert env.n_assets == 1


def test_stable_baselines3_compatibility():
    features = pd.DataFrame(np.random.randn(50, 5).astype(np.float32))
    prices = pd.DataFrame(np.random.uniform(50, 100, (50, 2)).astype(np.float32))

    env = IndiaMarketTradingEnv(features, prices)
    check_env(env)


def test_render():
    features = pd.DataFrame(np.random.randn(10, 2))
    prices = pd.DataFrame(np.random.uniform(50, 100, (10, 1)))
    env = IndiaMarketTradingEnv(features, prices, render_mode="human")
    env.reset()
    env.step(np.array([1.0]))
    # Verify render executes without exception
    env.render()


def test_include_cash_allocation():
    features = pd.DataFrame(np.zeros((10, 4)))
    # Single index price increasing from 100 to 200
    prices = pd.DataFrame(np.linspace(100.0, 200.0, 10).reshape(-1, 1))

    env = IndiaMarketTradingEnv(features, prices, include_cash=True)
    obs, info = env.reset()

    # Cash asset added: n_assets == 2 (Index and Cash)
    assert env.n_assets == 2
    assert obs.shape == (4 + 2,)
    assert env.action_space.shape == (2,)

    # Allocate 100% to Cash: [0.0, 1.0]
    obs, reward, term, trunc, info = env.step(np.array([0.0, 1.0]))
    # Keep holding 100% cash
    obs, reward, term, trunc, info = env.step(np.array([0.0, 1.0]))
    # Turnover is 0 and cash return is 0
    assert np.isclose(info["turnover"], 0.0)
    assert np.isclose(info["portfolio_return"], 0.0)

    # SB3 compatibility with include_cash=True
    check_env(env)


def test_performance_metrics_calculation():
    from envs.metrics import calculate_performance_metrics

    # Test portfolio values and returns
    values = np.array([100.0, 105.0, 110.0, 100.0, 120.0])
    returns = np.diff(values) / values[:-1]

    metrics = calculate_performance_metrics(values, returns)
    assert np.isclose(metrics["cumulative_return_pct"], 20.0)
    # Peak is 110, trough is 100 -> drawdown is (100 - 110) / 110 = -9.09% -> MDD = 9.09%
    assert np.isclose(metrics["max_drawdown_pct"], (10.0 / 110.0) * 100.0, rtol=1e-3)
    assert np.isfinite(metrics["annualized_sharpe"])
    assert np.isfinite(metrics["annualized_sortino"])
