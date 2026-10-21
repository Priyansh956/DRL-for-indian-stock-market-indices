import numpy as np
import pandas as pd
from envs.india_trading_env import IndiaMarketTradingEnv

def test_environment_initialization():
    mock_features = pd.DataFrame(np.random.randn(100, 10))
    mock_prices = pd.DataFrame(100 + np.cumsum(np.random.randn(100, 3) * 0.1, axis=0))
    
    env = IndiaMarketTradingEnv(mock_features, mock_prices)
    obs, info = env.reset()
    
    # Check observation shape: features + portfolio weights
    assert obs.shape == (10 + 3,)
    assert np.isclose(np.sum(env.weights), 1.0)

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
    assert not terminated