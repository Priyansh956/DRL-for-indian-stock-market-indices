# Architecture

## Purpose and scope

The application is a single-process, historical reinforcement-learning backtest for a portfolio allocating between the NIFTY 50 index and a zero-return cash proxy. It is organized around a data-preparation module, a Gymnasium simulation environment, and a command-line training/evaluation script.

It does not contain a live market feed, broker integration, order management, or deployment service.

## Component map

```text
Yahoo Finance
    │
    ▼
DataPreprocessor (data/data_preprocessing.py)
    ├── downloads ^NSEI OHLC data and ^INDIAVIX
    ├── aligns VIX with index dates and drops missing rows
    ├── calculates technical indicators
    └── chronological split + train-only feature normalization
           │
           ├── train features + adjusted-close prices
           │         │
           │         ▼
           │   IndiaMarketTradingEnv (envs/india_trading_env.py)
           │         │  Gymnasium observations, actions, rewards
           │         ▼
           │   Stable-Baselines3 PPO (train.py)
           │
           └── test features + adjusted-close prices
                     │
                     ▼
               deterministic policy evaluation
                     │
                     ├── buy-and-hold comparison
                     ├── metrics (envs/metrics.py)
                     └── results/portfolio_performance.png

Trained policy ──► models/ppo_nifty.zip
```

## Modules

### `data/data_preprocessing.py`

`DataPreprocessor` is responsible for retrieving the two Yahoo Finance series, feature engineering, and preparing aligned inputs for the environment.

`fetch_data()` downloads NIFTY 50 (`^NSEI`) and India VIX (`^INDIAVIX`) for the configured date range. It flattens yfinance multi-index columns when present, adds VIX's adjusted close as the `VIX` column, and drops rows missing either series.

`compute_technical_indicators()` calculates:

- Log returns from adjusted close.
- Simple moving averages over 20 and 50 rows.
- RSI using a 14-row rolling average of gains and losses.
- MACD (12-period EMA minus 26-period EMA) and its 9-period EMA signal.
- ATR over 14 rows from high, low, and the previous adjusted close.
- Annualized rolling volatility from 20-row log-return standard deviation and a `sqrt(250)` factor.

Rows with missing indicator values are then dropped. The model's feature list is RSI, MACD, MACD signal, ATR, rolling volatility, and VIX; log returns and the simple moving averages are computed but are not policy inputs.

`get_train_test_data()` splits chronologically at the requested date: training indices are before the split and test indices are on/after it. Feature means and standard deviations are learned from training rows only and reused to normalize test rows. The return value is `((train_features, train_prices), (test_features, test_prices))`. Price frames contain adjusted close and retain their dates as the index.

`get_processed_data()` is a convenience API for one processed period. Unlike the split-aware method, it normalizes across the entire period, so it should not be used to produce train/test backtest inputs where leakage matters.

### `envs/india_trading_env.py`

`IndiaMarketTradingEnv` implements the Gymnasium environment consumed by PPO.

- **Observation:** Current feature row concatenated with current portfolio weights.
- **Action:** A continuous target allocation vector. Negative entries are clipped and non-zero actions are normalized to sum to one. Invalid or all-zero vectors use a safe fallback.
- **Assets:** With `include_cash=True` and one input price column, a constant price-1.0 cash column is added. The training script enables this, creating NIFTY and cash allocations.
- **Transition:** A step moves from the current price row to the next row, and the target allocation is applied to that next-period return.
- **Trading friction:** Turnover is the sum of absolute allocation changes. The transaction-cost amount is turnover multiplied by the configured cost percentage and current portfolio value.
- **Reward:** For the first `reward_window - 1` returns, reward is the current net return. Once enough returns exist, reward is the mean divided by standard deviation of the rolling returns, clipped to `[-10, 10]`; zero volatility is handled separately.
- **Episode:** Ends at the final available price row or if portfolio value reaches zero.

The environment starts with a 100% allocation to the first asset. This means the first transition from the default allocation to a new target can itself incur turnover/cost. The cash proxy has no yield and is not a model of a real cash product.

### `train.py`

The command-line entry point orchestrates the run:

1. Parse dates, PPO timesteps, seed, and output directories.
2. Build the split-aware training/test datasets.
3. Train PPO with a multilayer perceptron policy in a `DummyVecEnv`.
4. Save the model as `ppo_nifty.zip`.
5. Evaluate the learned policy deterministically over the held-out test rows.
6. Simulate a buy-and-hold NIFTY baseline on the same test period.
7. Print metrics and save a portfolio/drawdown/allocation figure.

The PPO configuration currently uses a learning rate of `3e-4`, 1,024 rollout steps, batch size 64, 10 epochs, `gamma=0.99`, `gae_lambda=0.95`, clipping range `0.2`, and entropy coefficient `0.01`.

Evaluation uses the trained policy's actions but does not update the policy. Train and test environments each start with the default 100,000 units of capital.

### `envs/metrics.py`

This module computes cumulative return, maximum drawdown, annualized Sharpe ratio, and annualized Sortino ratio. The risk-free rate defaults to zero and annualization uses 252 periods per year. The reward used for learning is distinct from these final evaluation metrics.

## Runtime data and outputs

The default dates and output paths are set by `train.py` command-line arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--start-date` | `2018-01-01` | Inclusive yfinance download start. |
| `--end-date` | `2026-01-01` | Exclusive yfinance download end. |
| `--split-date` | `2024-01-01` | First date assigned to the test period. |
| `--timesteps` | `30000` | PPO learning timesteps. |
| `--seed` | `42` | PPO/environment seed. |
| `--models-dir` | `models` | Directory for the saved PPO model. |
| `--results-dir` | `results` | Directory for the evaluation plot. |

The expected outputs are `<models-dir>/ppo_nifty.zip` and `<results-dir>/portfolio_performance.png`. Output directories are created if missing.

The data pipeline requires network access to Yahoo Finance. The environment itself consumes in-memory feature and price arrays and is tested separately with synthetic data.

## Testing and local execution

The repository CI uses Python 3.12, installs `requirements.txt` and `requirements-dev.txt`, installs the package in editable mode, runs Ruff, then runs pytest.

For setup and copy-paste run commands, including a short smoke run, see [README.md](./README.md).
