import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.getcwd())
from envs.india_trading_env import IndiaMarketTradingEnv


def calculate_indicators(df):
    """Calculates the 6 technical indicators needed for the DRL state."""
    delta = df["Adj Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["Adj Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Adj Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Adj Close"])
    low_close = np.abs(df["Low"] - df["Adj Close"])
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    df["Rolling_Vol"] = df["Adj Close"].pct_change().rolling(20).std() * np.sqrt(252)
    return df


def calculate_drawdown(values):
    """Calculates the drawdown percentage from the peak."""
    values = np.array(values)
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    return drawdown * 100  # Return as percentage


def main():
    print("=" * 60)
    print("🚀 SORTINO RATIO DRL PROTOTYPE (3-PANEL EVALUATION)")
    print("=" * 60)

    # 1. LOAD REAL DATA
    print("\n📥 Loading expanded market data (2015-2024)...")
    nifty_df = pd.read_csv("data/NSEI.csv", usecols=["Date", "Open", "High", "Low", "Close", "Adj Close"])
    nifty_df["Date"] = pd.to_datetime(nifty_df["Date"])
    nifty_df.set_index("Date", inplace=True)

    vix_df = pd.read_csv("data/INDIAVIX.csv", usecols=["Date", "Adj Close"])
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])
    vix_df.set_index("Date", inplace=True)
    vix_df.rename(columns={"Adj Close": "VIX"}, inplace=True)

    df = nifty_df.join(vix_df, how="left")
    df["VIX"] = df["VIX"].ffill().bfill()
    df = df.dropna()
    print(f"✅ Loaded {len(df)} days of real market data.")

    # 2. CALCULATE INDICATORS
    print("\n⚙️ Calculating technical indicators...")
    df = calculate_indicators(df)
    df = df.dropna()

    # 3. SPLIT DATA
    split_date = pd.to_datetime("2023-01-01")
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]

    feature_cols = ["RSI", "MACD", "MACD_Signal", "ATR", "Rolling_Vol", "VIX"]

    train_mean = train_df[feature_cols].mean()
    train_std = train_df[feature_cols].std() + 1e-8

    train_features = (train_df[feature_cols] - train_mean) / train_std
    test_features = (test_df[feature_cols] - train_mean) / train_std

    train_prices = train_df[["Adj Close"]]
    test_prices = test_df[["Adj Close"]]

    print(f"✅ Training data shape: {train_features.shape}")
    print(f"✅ Testing data shape: {test_features.shape}")

    # 4. TRAIN THE AI MODEL (Using Sortino Reward in Environment)
    print("\n🧠 Training PPO Agent (Optimized with Sortino Ratio)...")

    def make_env():
        return IndiaMarketTradingEnv(
            df_features=train_features,
            price_matrix=train_prices,
            transaction_cost_pct=0.001,
            initial_capital=100000.0,
            reward_window=60,
            include_cash=True,
        )

    env = DummyVecEnv([make_env])

    model = PPO(policy="MlpPolicy", env=env, learning_rate=1e-4, n_steps=2048, batch_size=128, n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=1)

    model.learn(total_timesteps=50000)
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_nifty_sortino")
    print("💾 Sortino model saved.")

    # 5. EVALUATE AND PLOT (3-PANEL)
    print("\n📊 Evaluating AI Agent on unseen Test Data...")

    def make_test_env():
        return IndiaMarketTradingEnv(
            df_features=test_features,
            price_matrix=test_prices,
            transaction_cost_pct=0.001,
            initial_capital=100000.0,
            reward_window=60,
            include_cash=True,
        )

    test_env = DummyVecEnv([make_test_env])
    obs = test_env.reset()

    portfolio_values = []
    weights_history = []  # To store allocation weights for Panel 3

    for i in range(len(test_features) - 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
        portfolio_values.append(info[0]["portfolio_value"])
        weights_history.append(info[0]["weights"])  # Assuming weights is [nifty_weight, cash_weight]

    # Baseline
    initial_price = test_prices.iloc[0]["Adj Close"]
    baseline_values = 100000.0 * (test_prices["Adj Close"] / initial_price)

    # Calculate Drawdowns
    drl_drawdown = calculate_drawdown(portfolio_values)
    bh_drawdown = calculate_drawdown(baseline_values.values)

    # Prepare weights for plotting
    weights_array = np.array(weights_history)
    nifty_weights = weights_array[:, 0]  # Assuming index 0 is NIFTY
    cash_weights = weights_array[:, 1]  # Assuming index 1 is Cash
    dates = test_features.index.tolist()[: len(portfolio_values)]

    # --- 3-PANEL PLOTTING ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.suptitle("NIFTY 50 Out-of-Sample Evaluation: DRL Agent (Sortino) vs. Buy & Hold", fontsize=16, fontweight="bold")

    # Panel 1: Portfolio Value
    ax1.plot(dates, portfolio_values, label="DRL Agent (PPO + Sortino)", color="#1f77b4", linewidth=2)
    ax1.plot(test_prices.index, baseline_values, label="Buy & Hold NIFTY 50", color="#ff7f0e", linestyle="--", linewidth=2)
    ax1.set_ylabel("Portfolio Value (₹)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Portfolio Growth", fontsize=12)

    # Panel 2: Drawdown (Risk)
    ax2.plot(dates, drl_drawdown, label="DRL Agent Drawdown", color="#1f77b4", linewidth=1.5)
    ax2.plot(test_prices.index, bh_drawdown, label="Buy & Hold Drawdown", color="#ff7f0e", linestyle="--", linewidth=1.5)
    ax2.fill_between(dates, drl_drawdown, 0, color="#1f77b4", alpha=0.2)
    ax2.fill_between(test_prices.index, bh_drawdown, 0, color="#ff7f0e", alpha=0.1)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Risk Analysis (Drawdown from Peak)", fontsize=12)

    # Panel 3: Allocation Weight
    ax3.stackplot(dates, nifty_weights, cash_weights, labels=["NIFTY 50 Weight", "Cash Proxy Weight"], colors=["#2ca02c", "#7f7f7f"], alpha=0.8)
    ax3.set_ylabel("Allocation Weight", fontsize=12)
    ax3.set_xlabel("Date", fontsize=12)
    ax3.legend(loc="upper left")
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("AI Asset Allocation Strategy", fontsize=12)

    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/sortino_3panel_evaluation.png", dpi=300)
    print("📈 3-Panel Plot saved to results/sortino_3panel_evaluation.png")
    plt.show()

    # Print Metrics
    drl_return = ((portfolio_values[-1] - 100000) / 100000) * 100
    bh_return = ((baseline_values.iloc[-1] - 100000) / 100000) * 100
    print("\n📊 FINAL RESULTS:")
    print(f"DRL Return: {drl_return:.2f}%")
    print(f"Buy & Hold Return: {bh_return:.2f}%")
    print("\n🎉 Sortino pipeline complete!")


if __name__ == "__main__":
    main()
