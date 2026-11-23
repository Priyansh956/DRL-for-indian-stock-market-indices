"""Custom Gymnasium trading environment for risk-adjusted portfolio management on Indian market indices."""

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


def calculate_indian_market_friction(
    buy_value: float,
    sell_value: float,
    stt_rate: float = 0.001,
    stamp_duty_rate: float = 0.00015,
    brokerage_rate: float = 0.0002,
    exchange_rate: float = 0.0000325,
    sebi_rate: float = 0.000001,
    gst_rate: float = 0.18,
) -> tuple[float, dict[str, float]]:
    """Calculates realistic asymmetric Indian market trading frictions.

    Components:
        - STT (Securities Transaction Tax): 0.1% on SELL turnover only (equity delivery).
        - Stamp Duty: 0.015% on BUY turnover only.
        - Brokerage: Commission on total turnover (buy + sell).
        - Exchange & SEBI charges: Regulatory fees on turnover.
        - GST: 18% on service charges (brokerage + exchange charges + SEBI charges).

    Args:
        buy_value: Total rupee value of asset purchases.
        sell_value: Total rupee value of asset sales.
        stt_rate: STT rate applied only to sell_value (default: 0.001 / 0.1%).
        stamp_duty_rate: Stamp duty rate applied only to buy_value (default: 0.00015 / 0.015%).
        brokerage_rate: Brokerage rate applied to total turnover (default: 0.0002 / 0.02%).
        exchange_rate: Exchange transaction fee rate (default: 0.0000325 / 0.00325%).
        sebi_rate: SEBI turnover fee rate (default: 0.000001 / 0.0001%).
        gst_rate: Goods and Services Tax rate applied to service charges (default: 0.18 / 18%).

    Returns:
        total_cost: Total rupee cost of all frictions.
        breakdown: Dictionary detailing each friction component.
    """
    buy_val = max(0.0, float(buy_value))
    sell_val = max(0.0, float(sell_value))
    turnover = buy_val + sell_val

    # Asymmetric regulatory taxes
    stt = sell_val * float(stt_rate)
    stamp_duty = buy_val * float(stamp_duty_rate)

    # Turnover service fees
    brokerage = turnover * float(brokerage_rate)
    exchange_fee = turnover * float(exchange_rate)
    sebi_fee = turnover * float(sebi_rate)

    # GST on service charges
    service_charges = brokerage + exchange_fee + sebi_fee
    gst = service_charges * float(gst_rate)

    total_cost = stt + stamp_duty + service_charges + gst

    breakdown = {
        "stt": float(stt),
        "stamp_duty": float(stamp_duty),
        "brokerage": float(brokerage),
        "exchange_fee": float(exchange_fee),
        "sebi_fee": float(sebi_fee),
        "gst": float(gst),
        "total_friction": float(total_cost),
    }
    return float(total_cost), breakdown


class IndiaMarketTradingEnv(gym.Env):
    """Custom Gymnasium Environment for Risk-Adjusted DRL Trading on Indian Indices.

    State:
        Technical indicators (features) + current portfolio allocation weights.

    Action:
        Continuous target portfolio weights. Normalized via simplex projection
        to ensure non-negative weights that sum to 1.0.

    Reward:
        Risk-adjusted reward based on a rolling Sharpe Ratio approximation.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        df_features: pd.DataFrame | np.ndarray,
        price_matrix: pd.DataFrame | np.ndarray,
        transaction_cost_pct: float = 0.001,
        initial_capital: float = 100000.0,
        reward_window: int = 20,
        include_cash: bool = False,
        use_indian_friction: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()

        # Ingest and validate features
        if isinstance(df_features, pd.DataFrame):
            self.df_features = df_features.to_numpy(dtype=np.float32)
        else:
            self.df_features = np.asarray(df_features, dtype=np.float32)

        if self.df_features.ndim == 1:
            self.df_features = self.df_features.reshape(-1, 1)

        # Ingest and validate prices
        if isinstance(price_matrix, pd.DataFrame):
            self.prices = price_matrix.to_numpy(dtype=np.float32)
        else:
            self.prices = np.asarray(price_matrix, dtype=np.float32)

        if self.prices.ndim == 1:
            self.prices = self.prices.reshape(-1, 1)

        # Support cash allocation (constant price proxy) for single-asset index trading
        self.include_cash = include_cash
        if self.include_cash and self.prices.shape[1] == 1:
            cash_col = np.full((len(self.prices), 1), 1.0, dtype=np.float32)
            self.prices = np.hstack([self.prices, cash_col])

        if len(self.df_features) != len(self.prices):
            raise ValueError(f"Feature rows ({len(self.df_features)}) must match price rows ({len(self.prices)}).")

        if len(self.df_features) < 2:
            raise ValueError(f"Dataset must contain at least 2 rows for transitions, got {len(self.df_features)}.")

        self.n_steps, self.n_features = self.df_features.shape
        self.n_assets = self.prices.shape[1]
        self.tc = float(transaction_cost_pct)
        self.initial_capital = float(initial_capital)
        self.reward_window = int(reward_window)
        self.use_indian_friction = bool(use_indian_friction)
        self.render_mode = render_mode

        # Action space: Portfolio weights (Continuous simplex in [0, 1])
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        # State space: Technical indicators + current portfolio weights
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + self.n_assets,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_returns: list[float] = []

        self.reset()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = float(self.initial_capital)

        # Default allocation: 100% in first asset (or cash proxy)
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.weights[0] = 1.0
        self.portfolio_returns = []

        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.copy(),
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        step_idx = min(self.current_step, self.n_steps - 1)
        features = self.df_features[step_idx]
        obs = np.concatenate([features, self.weights]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Sanitizes and projects raw action vectors onto the probability simplex."""
        action_arr = np.asarray(action, dtype=np.float32).flatten()

        if action_arr.shape != (self.n_assets,) or not np.all(np.isfinite(action_arr)):
            # Fallback to current weights if valid, or uniform allocation
            if np.isclose(np.sum(self.weights), 1.0):
                return self.weights.copy()
            return np.ones(self.n_assets, dtype=np.float32) / self.n_assets

        # Non-negative projection
        clipped = np.maximum(action_arr, 0.0)
        action_sum = float(np.sum(clipped))

        if action_sum > 1e-8:
            return (clipped / action_sum).astype(np.float32)

        # Fallback to equal weighting if all values are zero or negative
        return np.ones(self.n_assets, dtype=np.float32) / self.n_assets

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Episode already terminated check
        if self.current_step >= self.n_steps - 1 or self.portfolio_value <= 0.0:
            obs = self._get_observation()
            info = {
                "step": self.current_step,
                "portfolio_value": self.portfolio_value,
                "portfolio_return": 0.0,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "weights": self.weights.copy(),
                "reward": 0.0,
            }
            return obs, 0.0, True, False, info

        # Project action onto simplex
        target_weights = self._normalize_action(action)

        # Calculate turnover and transaction cost friction
        turnover = float(np.sum(np.abs(target_weights - self.weights)))

        if self.use_indian_friction:
            # When cash is included, cash asset (last column) is exempt from STT/stamp duty
            risky_end = (self.n_assets - 1) if self.include_cash else self.n_assets
            risky_delta = target_weights[:risky_end] - self.weights[:risky_end]
            buy_val = float(np.sum(np.maximum(risky_delta, 0.0))) * self.portfolio_value
            sell_val = float(np.sum(np.maximum(-risky_delta, 0.0))) * self.portfolio_value
            cost, friction_breakdown = calculate_indian_market_friction(buy_val, sell_val)
        else:
            cost = turnover * self.tc * self.portfolio_value
            friction_breakdown = {}

        # Move to next price step
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        # Calculate asset returns
        price_prev = self.prices[self.current_step - 1]
        price_curr = self.prices[self.current_step]
        price_prev_safe = np.where(np.abs(price_prev) < 1e-8, 1e-8, price_prev)
        asset_returns = (price_curr - price_prev) / price_prev_safe

        # Net portfolio return after transaction cost
        gross_return = float(np.sum(target_weights * asset_returns))
        cost_ratio = (cost / self.portfolio_value) if self.portfolio_value > 0 else 0.0
        portfolio_return = gross_return - cost_ratio

        # Update portfolio value
        self.portfolio_value *= 1.0 + portfolio_return

        # Bankruptcy condition
        if self.portfolio_value <= 0.0:
            self.portfolio_value = 0.0
            terminated = True

        self.portfolio_returns.append(portfolio_return)
        self.weights = target_weights

        # Risk-adjusted reward (rolling Sharpe ratio approximation)
        reward = self._calculate_reward(portfolio_return)

        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "portfolio_value": float(self.portfolio_value),
            "portfolio_return": float(portfolio_return),
            "turnover": float(turnover),
            "transaction_cost": float(cost),
            "weights": self.weights.copy(),
            "reward": float(reward),
        }
        if self.use_indian_friction:
            info["friction_breakdown"] = friction_breakdown

        return obs, float(reward), terminated, truncated, info

    def _calculate_reward(self, current_return: float) -> float:
        """Calculates risk-adjusted reward based on an IMPROVISED rolling Sortino ratio."""
        if len(self.portfolio_returns) >= self.reward_window:
            recent_returns = np.array(self.portfolio_returns[-self.reward_window :], dtype=np.float64)
            mean_return = float(np.mean(recent_returns))

            # --- SHARPE RATIO (COMMENTED OUT) ---
            # std = float(np.std(recent_returns))
            # if std < 1e-8: return float(np.clip(mean_return, -1.0, 1.0))
            # sharpe = mean_return / std
            # return float(np.clip(sharpe, -10.0, 10.0))

            # --- IMPROVISED SORTINO RATIO (ACTIVE) ---
            # 1. Define Minimum Acceptable Return (MAR).
            # In finance, MAR is often the risk-free rate, but 0 is standard for daily DRL.
            mar = 0.0

            # 2. Calculate Downside Deviations (How much each day fell below MAR)
            # We use np.maximum to isolate only the negative deviations
            downside_deviations = np.maximum(mar - recent_returns, 0.0)

            # 3. Calculate Semi-Deviation (The mathematically correct Sortino denominator)
            # Formula: sqrt( mean( downside_deviations^2 ) )
            squared_downside = downside_deviations**2
            semi_deviation = float(np.sqrt(np.mean(squared_downside)))

            # 4. Smoothing Factor (Epsilon)
            # This is the "Improvisation". It prevents division by zero and stops
            # the reward from exploding during low-volatility bull markets.
            epsilon = 1e-4

            # 5. Calculate Sortino
            sortino = mean_return / (semi_deviation + epsilon)

            # 6. Clip the reward to stabilize PPO neural network training
            return float(np.clip(sortino, -5.0, 5.0))

        # Insufficient return history: use immediate net return
        return float(current_return)

    def render(self) -> None:
        if self.render_mode == "human" or self.render_mode is None:
            print(f"Step: {self.current_step} | Portfolio Value: ₹{self.portfolio_value:,.2f} | Weights: {np.round(self.weights, 4)}")
