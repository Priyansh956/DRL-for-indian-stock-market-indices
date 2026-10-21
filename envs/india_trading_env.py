import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class IndiaMarketTradingEnv(gym.Env):
    """
    Custom Gym Environment for Risk-Adjusted DRL Trading on Indian Indices.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df_features: pd.DataFrame,
        price_matrix: pd.DataFrame,
        transaction_cost_pct: float = 0.001,
    ):
        super(IndiaMarketTradingEnv, self).__init__()

        self.df_features = df_features.values
        self.prices = price_matrix.values
        self.n_steps, self.n_features = self.df_features.shape
        self.n_assets = self.prices.shape[1]
        self.tc = transaction_cost_pct

        # Action space: Portfolio weights summing to 1 (Continuous simplex)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        # State space: Technical indicators + current portfolio weights
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features + self.n_assets,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = 100000.0  # Initial capital in INR (₹)
        self.weights = np.zeros(self.n_assets)
        self.weights[0] = 1.0  # Default to cash or first asset
        self.portfolio_returns = []

        return self._get_observation(), {}

    def _get_observation(self):
        features = self.df_features[self.current_step]
        return np.concatenate([features, self.weights]).astype(np.float32)

    def step(self, action):
        # Normalize weights to sum to 1 (Simplex projection)
        target_weights = np.clip(action, 0, 1)
        if np.sum(target_weights) > 0:
            target_weights /= np.sum(target_weights)
        else:
            target_weights = np.ones(self.n_assets) / self.n_assets

        # Calculate turnover (transaction friction penalty)
        turnover = np.sum(np.abs(target_weights - self.weights))
        cost = turnover * self.tc * self.portfolio_value

        # Move to next step prices
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # Calculate asset returns
        price_prev = self.prices[self.current_step - 1]
        price_curr = self.prices[self.current_step]
        asset_returns = (price_curr - price_prev) / price_prev

        # Portfolio return minus transaction cost penalty
        portfolio_return = np.sum(target_weights * asset_returns) - (cost / self.portfolio_value)
        self.portfolio_value *= 1 + portfolio_return
        self.portfolio_returns.append(portfolio_return)

        self.weights = target_weights

        # Risk-Adjusted Reward Design (Rolling Sharpe Ratio approximation)
        if len(self.portfolio_returns) > 20:
            recent_returns = np.array(self.portfolio_returns[-20:])
            vol = np.std(recent_returns) + 1e-8
            reward = np.mean(recent_returns) / vol  # Instantaneous Sharpe Ratio reward
        else:
            reward = portfolio_return

        truncated = False
        return (
            self._get_observation(),
            float(reward),
            done,
            truncated,
            {"portfolio_value": self.portfolio_value},
        )

    def render(self):
        print(f"Step: {self.current_step} | Portfolio Value (₹): ₹{self.portfolio_value:,.2f}")
