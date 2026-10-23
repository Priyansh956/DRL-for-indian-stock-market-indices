"""PPO Training and Evaluation Pipeline for Risk-Adjusted Trading on Indian Market Indices."""

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from data.data_preprocessing import DataPreprocessor
from envs.india_trading_env import IndiaMarketTradingEnv
from envs.metrics import calculate_performance_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a PPO agent for NIFTY 50 index trading.")
    parser.add_argument("--start-date", type=str, default="2018-01-01", help="Dataset start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default="2026-01-01", help="Dataset end date (YYYY-MM-DD).")
    parser.add_argument("--split-date", type=str, default="2024-01-01", help="Chronological split date (YYYY-MM-DD).")
    parser.add_argument("--timesteps", type=int, default=30000, help="Total training timesteps for PPO.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save trained models.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save evaluation plots.")
    return parser.parse_args()


def setup_directories(models_dir: str, results_dir: str) -> tuple[Path, Path]:
    m_path = Path(models_dir)
    r_path = Path(results_dir)
    m_path.mkdir(parents=True, exist_ok=True)
    r_path.mkdir(parents=True, exist_ok=True)
    return m_path, r_path


def train_ppo_agent(
    train_features: pd.DataFrame,
    train_prices: pd.DataFrame,
    total_timesteps: int = 30000,
    seed: int = 42,
) -> tuple[PPO, DummyVecEnv]:
    """Instantiates the training environment and trains the PPO agent."""
    print("\n🚀 Initializing Training Environment...")

    def make_env():
        return IndiaMarketTradingEnv(
            df_features=train_features,
            price_matrix=train_prices,
            transaction_cost_pct=0.001,
            initial_capital=100000.0,
            reward_window=20,
            include_cash=True,
        )

    env = DummyVecEnv([make_env])
    env.seed(seed)

    print(f"🧠 Training PPO agent for {total_timesteps:,} timesteps (seed={seed})...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)
    print("✅ Training complete!")
    return model, env


def evaluate_agent(
    model: PPO,
    test_features: pd.DataFrame,
    test_prices: pd.DataFrame,
) -> dict[str, Any]:
    """Runs a deterministic evaluation of the trained agent on unseen test data."""
    print("\n🔍 Evaluating trained agent on unseen out-of-sample test data...")

    eval_env = IndiaMarketTradingEnv(
        df_features=test_features,
        price_matrix=test_prices,
        transaction_cost_pct=0.001,
        initial_capital=100000.0,
        reward_window=20,
        include_cash=True,
    )

    obs, _ = eval_env.reset()
    dates = test_features.index.tolist()

    history: dict[str, list] = {
        "dates": [dates[0]],
        "portfolio_values": [eval_env.portfolio_value],
        "portfolio_returns": [0.0],
        "nifty_weights": [float(eval_env.weights[0])],
        "cash_weights": [float(eval_env.weights[1])],
        "turnover": [0.0],
    }

    step_idx = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

        step_idx += 1
        current_date = dates[min(step_idx, len(dates) - 1)]

        history["dates"].append(current_date)
        history["portfolio_values"].append(info["portfolio_value"])
        history["portfolio_returns"].append(info["portfolio_return"])
        history["nifty_weights"].append(float(eval_env.weights[0]))
        history["cash_weights"].append(float(eval_env.weights[1]))
        history["turnover"].append(info["turnover"])

    return history


def compute_buy_and_hold_baseline(
    test_prices: pd.DataFrame,
    initial_capital: float = 100000.0,
    tc_pct: float = 0.001,
) -> dict[str, Any]:
    """Simulates a passive Buy-and-Hold NIFTY 50 strategy over the exact test period."""
    prices = test_prices["Adj Close"].values
    dates = test_prices.index.tolist()

    # Initial purchase incurs transaction cost: V0 = initial_capital * (1 - tc)
    starting_value = initial_capital * (1.0 - tc_pct)
    bh_values = [starting_value]
    bh_returns = [0.0]

    for t in range(1, len(prices)):
        day_return = (prices[t] - prices[t - 1]) / prices[t - 1]
        new_val = bh_values[-1] * (1.0 + day_return)
        bh_values.append(new_val)
        bh_returns.append(day_return)

    return {
        "dates": dates,
        "portfolio_values": bh_values,
        "portfolio_returns": bh_returns,
    }


def plot_performance(
    agent_history: dict[str, Any],
    bh_history: dict[str, Any],
    output_path: Path,
) -> None:
    """Generates and saves a demonstration performance comparison figure."""
    dates = pd.to_datetime(agent_history["dates"])
    agent_vals = np.array(agent_history["portfolio_values"])
    bh_vals = np.array(bh_history["portfolio_values"])

    # Drawdown series
    agent_peaks = np.maximum.accumulate(agent_vals)
    agent_dd = (agent_vals - agent_peaks) / agent_peaks * 100.0

    bh_peaks = np.maximum.accumulate(bh_vals)
    bh_dd = (bh_vals - bh_peaks) / bh_peaks * 100.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]})

    # 1. Portfolio Value Comparison
    axes[0].plot(dates, agent_vals, label="DRL Agent (PPO + Risk-Adjusted)", color="#1f77b4", linewidth=2.0)
    axes[0].plot(dates, bh_vals, label="Buy & Hold NIFTY 50", color="#ff7f0e", linestyle="--", linewidth=1.8)
    axes[0].set_ylabel("Portfolio Value (₹)", fontsize=11, fontweight="bold")
    axes[0].set_title("NIFTY 50 Out-of-Sample Evaluation: DRL Agent vs. Buy & Hold", fontsize=14, fontweight="bold", pad=12)
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"₹{x:,.0f}"))

    # 2. Drawdown Comparison
    axes[1].plot(dates, agent_dd, label="DRL Agent Drawdown", color="#1f77b4", linewidth=1.5)
    axes[1].plot(dates, bh_dd, label="Buy & Hold Drawdown", color="#ff7f0e", linestyle="--", linewidth=1.5)
    axes[1].fill_between(dates, agent_dd, 0, color="#1f77b4", alpha=0.15)
    axes[1].set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    axes[1].legend(loc="lower left", frameon=True)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # 3. Dynamic Asset Allocation (NIFTY vs Cash)
    axes[2].stackplot(
        dates,
        agent_history["nifty_weights"],
        agent_history["cash_weights"],
        labels=["NIFTY 50 Weight", "Cash Proxy Weight"],
        colors=["#2ca02c", "#7f7f7f"],
        alpha=0.65,
    )
    axes[2].set_ylabel("Allocation Weight", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Date", fontsize=11, fontweight="bold")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(loc="center left", frameon=True)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"📊 Performance plot successfully saved to: {output_path}")


def main() -> None:
    args = parse_args()
    models_dir, results_dir = setup_directories(args.models_dir, args.results_dir)

    print("=" * 70)
    print("📈 NIFTY 50 RISK-ADJUSTED DRL TRADING PIPELINE (PPO PROTOTYPE)")
    print("=" * 70)

    # 1. Data Ingestion & Chronological Split
    preprocessor = DataPreprocessor(
        ticker="^NSEI",
        vix_ticker="^INDIAVIX",
        start_date=args.start_date,
        end_date=args.end_date,
    )
    (train_features, train_prices), (test_features, test_prices) = preprocessor.get_train_test_data(split_date=args.split_date)
    tr_start = train_features.index[0].strftime("%Y-%m-%d")
    tr_end = train_features.index[-1].strftime("%Y-%m-%d")
    te_start = test_features.index[0].strftime("%Y-%m-%d")
    te_end = test_features.index[-1].strftime("%Y-%m-%d")
    print(f"📅 Train Period: {tr_start} to {tr_end} ({len(train_features)} days)")
    print(f"📅 Test Period:  {te_start} to {te_end} ({len(test_features)} days)")

    # 2. Train PPO Agent
    model, _ = train_ppo_agent(
        train_features=train_features,
        train_prices=train_prices,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )

    # Save model
    model_path = models_dir / "ppo_nifty.zip"
    model.save(str(model_path))
    print(f"💾 Trained model saved to: {model_path}")

    # 3. Evaluate Agent on Unseen Test Data
    agent_history = evaluate_agent(model, test_features, test_prices)

    # 4. Baseline Comparison (Buy & Hold NIFTY 50)
    bh_history = compute_buy_and_hold_baseline(test_prices, initial_capital=100000.0)

    # 5. Performance Metrics
    agent_metrics = calculate_performance_metrics(
        portfolio_values=np.array(agent_history["portfolio_values"]),
        returns=np.array(agent_history["portfolio_returns"][1:]),
    )
    bh_metrics = calculate_performance_metrics(
        portfolio_values=np.array(bh_history["portfolio_values"]),
        returns=np.array(bh_history["portfolio_returns"][1:]),
    )

    # 6. Report Results Table
    print("\n" + "=" * 70)
    print("📋 OUT-OF-SAMPLE EVALUATION RESULTS (Unseen Test Set)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'DRL Agent (PPO)':<18} | {'Buy & Hold NIFTY':<18}")
    print("-" * 72)
    print(f"{'Initial Capital (₹)':<30} | ₹{100000.0:,.2f}          | ₹{100000.0:,.2f}")
    print(f"{'Final Portfolio Value (₹)':<30} | ₹{agent_history['portfolio_values'][-1]:,.2f}          | ₹{bh_history['portfolio_values'][-1]:,.2f}")
    print(f"{'Cumulative Return (%)':<30} | {agent_metrics['cumulative_return_pct']:>16.2f}% | {bh_metrics['cumulative_return_pct']:>16.2f}%")
    print(f"{'Maximum Drawdown (%)':<30} | {agent_metrics['max_drawdown_pct']:>16.2f}% | {bh_metrics['max_drawdown_pct']:>16.2f}%")
    print(f"{'Annualized Sharpe Ratio':<30} | {agent_metrics['annualized_sharpe']:>17.2f} | {bh_metrics['annualized_sharpe']:>17.2f}")
    print(f"{'Annualized Sortino Ratio':<30} | {agent_metrics['annualized_sortino']:>17.2f} | {bh_metrics['annualized_sortino']:>17.2f}")
    print("=" * 70)

    # 7. Generate Demonstration Plot
    plot_path = results_dir / "portfolio_performance.png"
    plot_performance(agent_history, bh_history, plot_path)


if __name__ == "__main__":
    main()
