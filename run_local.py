import os
import sys

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure local imports work
sys.path.insert(0, os.getcwd())

from data.data_preprocessing import DataPreprocessor
from envs.india_trading_env import IndiaMarketTradingEnv


def main():
    # 1. DATA LOADING
    print("\nFetching real market data via Yahoo Finance...")
    try:
        # Using NSEI and INDIAVIX without '^' sometimes bypasses local ISP blocks
        preprocessor = DataPreprocessor(ticker="^NSEI", vix_ticker="^INDIAVIX", start_date="2019-01-01", end_date="2024-01-01")
        (train_features, train_prices), (test_features, test_prices) = preprocessor.get_train_test_data(split_date="2023-01-01")
        print("Real data loaded successfully!")
    except Exception as e:
        print(f"yfinance failed: {e}")
        # print("Tip: If this keeps failing, manually download the CSVs from Yahoo Finance and place them in the 'data/' folder.")
        return

    # 2. ENVIRONMENT SETUP
    print("\nSetting up Gymnasium Trading Environment...")

    def make_env():
        return IndiaMarketTradingEnv(
            df_features=train_features,
            price_matrix=train_prices,
            transaction_cost_pct=0.001,  # 0.1% transaction cost to prevent overtrading
            initial_capital=100000.0,
            reward_window=20,  # Rolling Sharpe ratio window
            include_cash=True,
        )

    env = DummyVecEnv([make_env])

    # 3. AI MODEL TRAINING (PPO)
    print("\nTraining PPO Agent (This will take ~1-2 minutes)...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,  # Shows live training progress
    )

    # Train for 5,000 timesteps (Perfect for a quick prototype proof-of-concept)
    model.learn(total_timesteps=5000)

    # Save the trained AI model
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_nifty_local_prototype")
    print("Trained model saved to: models/ppo_nifty_local_prototype.zip")

    # 4. EVALUATION & PLOTTING (The "Money Shot" for your presentation)
    print("\nEvaluating AI Agent on unseen Test Data...")

    def make_test_env():
        return IndiaMarketTradingEnv(
            df_features=test_features,
            price_matrix=test_prices,
            transaction_cost_pct=0.001,
            initial_capital=100000.0,
            reward_window=20,
            include_cash=True,
        )

    test_env = DummyVecEnv([make_test_env])

    # Run the trained agent on the test environment
    obs = test_env.reset()
    portfolio_values = []

    # Simulate the test period step-by-step
    for i in range(len(test_features) - 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
        # DummyVecEnv returns info as a list of dictionaries
        portfolio_values.append(info[0]["portfolio_value"])

    # Calculate Buy & Hold Baseline
    initial_price = test_prices.iloc[0]["Adj Close"]
    baseline_values = 100000.0 * (test_prices["Adj Close"] / initial_price)

    # Plotting
    plt.figure(figsize=(14, 7))
    dates = test_features.index.tolist()[: len(portfolio_values)]

    plt.plot(dates, portfolio_values, label="DRL PPO Agent", color="#1f77b4", linewidth=2)
    plt.plot(test_prices.index, baseline_values, label="Buy & Hold Baseline", color="#ff7f0e", linestyle="--", linewidth=2)

    plt.title("DRL Agent vs. Buy & Hold Baseline (Test Period)", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Value (INR)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plot_path = "results/drl_performance_local.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Evaluation plot saved to: {plot_path}")
    plt.show()

    print("\nLocal prototype pipeline complete! Ready for presentation.")


if __name__ == "__main__":
    main()
