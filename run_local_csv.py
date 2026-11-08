import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.getcwd())
from envs.india_trading_env import IndiaMarketTradingEnv

def calculate_indicators(df):
    """Calculates the 6 technical indicators needed for the DRL state."""
    # 1. RSI (14 periods)
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. MACD and Signal Line
    exp1 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 3. ATR (14 periods)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Adj Close'])
    low_close = np.abs(df['Low'] - df['Adj Close'])
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # 4. Rolling Volatility (20 periods, annualized)
    df['Rolling_Vol'] = df['Adj Close'].pct_change().rolling(20).std() * np.sqrt(252)

    return df

def main():
    print("="*60)
    print("🚀 LOCAL DRL PROTOTYPE (REAL CSV DATA)")
    print("="*60)

    # 1. LOAD REAL DATA FROM CSV (Ignoring useless Dividend/Stock Split columns)
    print("\n📥 Loading real market data from local CSV files...")
    
    # Load only the columns we actually need for trading and indicators
    nifty_df = pd.read_csv('data/NSEI.csv', usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close'])
    nifty_df['Date'] = pd.to_datetime(nifty_df['Date'])
    nifty_df.set_index('Date', inplace=True)
    
    # Load only Date and Adj Close for VIX
    vix_df = pd.read_csv('data/INDIAVIX.csv', usecols=['Date', 'Adj Close'])
    vix_df['Date'] = pd.to_datetime(vix_df['Date'])
    vix_df.set_index('Date', inplace=True)
    vix_df.rename(columns={'Adj Close': 'VIX'}, inplace=True)

    # Merge VIX into Nifty dataframe (aligns by Date index automatically)
    df = nifty_df.join(vix_df, how='left')
    
    # Forward fill and backfill VIX to handle holidays/weekends where VIX data might be missing
    df['VIX'] = df['VIX'].ffill().bfill()
    
    # Drop any remaining NaNs (this will now only drop rows if High/Low/Close are actually missing)
    df = df.dropna()

    print(f"✅ Successfully loaded {len(df)} days of real market data.")

    # 2. CALCULATE INDICATORS
    print("\n⚙️ Calculating technical indicators...")
    df = calculate_indicators(df)
    df = df.dropna() # Drop initial NaN rows from indicator calculations (approx first 30 days)

    # 3. SPLIT DATA (Train: 2019-2022, Test: 2023)
    split_date = pd.to_datetime("2023-01-01")
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]

    feature_cols = ["RSI", "MACD", "MACD_Signal", "ATR", "Rolling_Vol", "VIX"]

    # Normalize features based on TRAIN data ONLY (Prevents look-ahead bias!)
    train_mean = train_df[feature_cols].mean()
    train_std = train_df[feature_cols].std() + 1e-8

    train_features = (train_df[feature_cols] - train_mean) / train_std
    test_features = (test_df[feature_cols] - train_mean) / train_std

    train_prices = train_df[["Adj Close"]]
    test_prices = test_df[["Adj Close"]]

    print(f"✅ Training data shape: {train_features.shape}")
    print(f"✅ Testing data shape: {test_features.shape}")

    # 4. TRAIN THE AI MODEL (PPO)
    print("\n🧠 Training PPO Agent (This will take ~1-2 minutes)...")
    def make_env():
        return IndiaMarketTradingEnv(
            df_features=train_features,
            price_matrix=train_prices,
            transaction_cost_pct=0.001,
            initial_capital=100000.0,
            reward_window=20,
            include_cash=True
        )

    env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )

    model.learn(total_timesteps=5000)

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_nifty_local_prototype")
    print("💾 Trained model saved to: models/ppo_nifty_local_prototype.zip")

    # 5. EVALUATE AND PLOT
    print("\n📊 Evaluating AI Agent on unseen Test Data...")
    def make_test_env():
        return IndiaMarketTradingEnv(
            df_features=test_features,
            price_matrix=test_prices,
            transaction_cost_pct=0.001,
            initial_capital=100000.0,
            reward_window=20,
            include_cash=True
        )

    test_env = DummyVecEnv([make_test_env])
    obs = test_env.reset()
    portfolio_values = []
    
    for i in range(len(test_features) - 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = test_env.step(action)
        portfolio_values.append(info[0]['portfolio_value'])

    # Baseline
    initial_price = test_prices.iloc[0]['Adj Close']
    baseline_values = 100000.0 * (test_prices['Adj Close'] / initial_price)

    # Plot
    plt.figure(figsize=(14, 7))
    dates = test_features.index.tolist()[:len(portfolio_values)]
    plt.plot(dates, portfolio_values, label="DRL PPO Agent", color="#1f77b4", linewidth=2)
    plt.plot(test_prices.index, baseline_values, label="Buy & Hold Baseline", color="#ff7f0e", linestyle="--", linewidth=2)
    
    plt.title("DRL Agent vs. Buy & Hold (Real NIFTY 50 Data)", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Value (INR)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/drl_performance_real_data.png", dpi=300)
    print("📈 Plot saved to results/drl_performance_real_data.png")
    plt.show()

# Calculate Performance Metrics
    
    # Final Returns
    drl_final = portfolio_values[-1]
    bh_final = baseline_values.iloc[-1]
    drl_return = ((drl_final - 100000) / 100000) * 100
    bh_return = ((bh_final - 100000) / 100000) * 100
    
    print(f"\n📈 Cumulative Return:")
    print(f"   DRL Agent:    {drl_return:.2f}%")
    print(f"   Buy & Hold:   {bh_return:.2f}%")
    
    # Calculate Max Drawdown
    def calculate_max_drawdown(values):
        values = np.array(values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return np.max(drawdown) * 100
    
    drl_mdd = calculate_max_drawdown(portfolio_values)
    bh_mdd = calculate_max_drawdown(baseline_values.values)
    
    print(f"\n📉 Maximum Drawdown:")
    print(f"   DRL Agent:    {drl_mdd:.2f}%")
    print(f"   Buy & Hold:   {bh_mdd:.2f}%")
    
    # Calculate Sharpe Ratio (annualized)
    def calculate_sharpe(returns, risk_free_rate=0.06):
        returns = np.array(returns)
        excess_returns = returns - (risk_free_rate/252)  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    drl_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    bh_returns = baseline_values.pct_change().dropna().values
    
    drl_sharpe = calculate_sharpe(drl_returns)
    bh_sharpe = calculate_sharpe(bh_returns)
    
    print(f"\n Annualized Sharpe Ratio:")
    print(f"   DRL Agent:    {drl_sharpe:.3f}")
    print(f"   Buy & Hold:   {bh_sharpe:.3f}")
    
    print("\n🎉 Local prototype pipeline complete!")

if __name__ == "__main__":
    main()