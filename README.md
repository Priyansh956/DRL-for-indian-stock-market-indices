# NIFTY 50 DRL Trading Prototype

This repository is a research prototype for training a reinforcement-learning agent to allocate a simulated portfolio between the NIFTY 50 index and a cash proxy. It downloads historical index and India VIX data, builds technical features, trains a Proximal Policy Optimization (PPO) model, and evaluates it against a buy-and-hold NIFTY 50 baseline.

This is a historical backtest, not a live trading system. It does not connect to a broker, place orders, or model all real-world trading effects. Its results are experimental and are not financial advice.

## What the model does

The current workflow is:

1. Download NIFTY 50 (`^NSEI`) and India VIX (`^INDIAVIX`) history from Yahoo Finance.
2. Calculate technical indicators and align the VIX series with index dates. Rows with missing values are removed.
3. Split the resulting rows chronologically into training and test periods. Fit feature normalization on the training period only, then apply those same statistics to the test period.
4. Train a PPO agent in a custom Gymnasium environment using the training period.
5. Run the trained policy deterministically on the later, held-out test period.
6. Compare portfolio results with a buy-and-hold NIFTY 50 baseline, print performance metrics, and save a plot.

By default, the data starts on `2018-01-01`, the split is at `2024-01-01`, and training runs for 30,000 timesteps. Yahoo Finance's `end` date is exclusive, so the default `2026-01-01` end date does not include observations dated on or after that day. Change `--end-date` when you want a later data window.

## Features and concepts

The policy receives six standardized features at each step:

| Feature | Brief explanation |
| --- | --- |
| RSI | Relative Strength Index; summarizes recent upward versus downward price movement. |
| MACD | Difference between 12-period and 26-period exponential moving averages; a momentum/trend indicator. |
| MACD signal | 9-period exponential moving average of MACD, commonly used to contextualize MACD changes. |
| ATR | Average True Range over 14 periods; estimates recent price movement/range. |
| Rolling volatility | Annualized standard deviation of 20-period log returns, using 250 trading days per year. |
| VIX | India VIX level, used as a market-volatility/sentiment input. |

The preprocessor also calculates log returns and 20-/50-period simple moving averages, but those are not currently included in the six policy inputs. Price data is kept separately to value the simulated portfolio.

### Reinforcement-learning terms in this project

- **Observation (state):** The six features for the current row, concatenated with the current portfolio allocation weights.
- **Action:** A vector of target portfolio weights. The environment clips negative values and normalizes positive values so allocations are non-negative and sum to one.
- **Cash proxy:** With `include_cash=True`, the environment adds a second asset whose price is always 1.0. It represents cash with zero modeled return—not an interest-bearing account.
- **Transition:** The environment applies the chosen allocation to the return from the current price row to the next price row.
- **Transaction cost and turnover:** A 0.1% cost is charged in the training/evaluation configuration according to the change in allocation weights. Turnover is the sum of absolute changes in weights.
- **Reward:** Before the 20-return window is filled, reward is the current net portfolio return. Thereafter, it is the mean divided by standard deviation of the recent returns (a rolling Sharpe-like reward), clipped to a bounded range. This is an approximation, not the annualized evaluation Sharpe ratio.
- **PPO:** Proximal Policy Optimization, an on-policy policy-gradient algorithm provided by Stable-Baselines3. The project uses an `MlpPolicy` (a multilayer perceptron) to map observations to allocation actions.
- **Train/test split:** Training uses dates strictly before the split date; evaluation uses dates on or after it. Feature mean and standard deviation are calculated only from the training portion to avoid using test-period statistics during training.
- **Baseline:** Buy-and-hold invests in the index over the same test window, with an initial transaction-cost adjustment.

The reported metrics are cumulative return, maximum drawdown, annualized Sharpe ratio, and annualized Sortino ratio. The plot compares portfolio values, drawdowns, and the agent's NIFTY-versus-cash allocation over the test dates.

## Repository layout

```text
data/
  data_preprocessing.py       Yahoo Finance ingestion and feature engineering
envs/
  india_trading_env.py        Gymnasium portfolio simulation environment
  metrics.py                  Return and risk metric calculations
train.py                      PPO training, evaluation, reporting, and plotting
tests/
  test_data_pipeline.py       Data/preprocessing tests (requires Yahoo Finance access)
  test_env.py                 Environment tests (uses synthetic data)
models/                       Saved model outputs
results/                      Evaluation plot outputs
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the component relationships and runtime data flow.

## Requirements and installation

Python 3.12 is used in the repository's CI workflow. From the repository root, create a virtual environment, install the project dependencies, then install the development/test tools:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

If Python 3.12 is already the Python selected by your terminal, you can use `python` in place of `py -3.12` when creating the environment. The commands above use the virtual environment's interpreter directly, so activation is not required.

Yahoo Finance downloads need internet access. The data pipeline tests also download external market data; the environment tests do not.

## Run tests and lint

Run the synthetic-data environment tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_env.py -q
```

Run the data-pipeline tests (requires internet/Yahoo Finance availability):

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_data_pipeline.py -q
```

Run all tests and lint checks:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff check .
```

## Train and inspect outputs

For a short end-to-end smoke run, use a smaller timestep count and separate output directories to avoid overwriting the default model and plot:

```powershell
.\.venv\Scripts\python.exe train.py `
  --start-date 2018-01-01 `
  --end-date 2026-10-25 `
  --split-date 2024-01-01 `
  --timesteps 1024 `
  --models-dir models\local-smoke `
  --results-dir results\local-smoke
```

`--end-date` is exclusive. This example requests data up to but not including `2026-10-25`; choose an end date appropriate to your run. A short run verifies the workflow but is not evidence that the learned strategy is effective. The default is 30,000 timesteps.

The script prints the train/test date ranges, training progress, and a comparison of the agent and buy-and-hold metrics. On completion, inspect:

- `models\local-smoke\ppo_nifty.zip` — trained PPO model.
- `results\local-smoke\portfolio_performance.png` — out-of-sample comparison plot.

Use `--help` to see all available options:

```powershell
.\.venv\Scripts\python.exe train.py --help
```

## Current limitations

- The data source is Yahoo Finance; downloads can fail or change format when the remote service is unavailable. No downloaded dataset is currently bundled.
- `get_processed_data()` standardizes over its entire data window. The training script instead uses `get_train_test_data()`, which fits normalization on training rows only; prefer that split-aware method for backtests.
- The model sees a small indicator set, uses a single historical train/test split, and has no walk-forward validation or hyperparameter study.
- Cash is modeled as a constant-price asset with zero return. Taxes, slippage beyond the configured proportional transaction cost, liquidity, dividends, and order execution are not modeled.
- Reported backtest results depend on the selected period, downloaded data, random seed, and training duration. Historical performance does not guarantee future performance.
