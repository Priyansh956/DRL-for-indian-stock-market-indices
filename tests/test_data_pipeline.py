import numpy as np

from data.data_preprocessing import DataPreprocessor
from envs.india_trading_env import IndiaMarketTradingEnv


def test_data_pipeline():
    processor = DataPreprocessor(ticker="^NSEI", start_date="2023-01-01", end_date="2026-01-01")
    features, prices = processor.get_processed_data()

    # Assertions to ensure data integrity
    assert not features.empty, "Feature dataframe is empty"
    assert not prices.empty, "Price dataframe is empty"
    assert len(features) == len(prices), "Feature and price lengths must match"
    print("✅ Data Ingestion and Feature Engineering verified successfully!")
    print(f"Total processed rows: {len(features)}")


def test_data_pipeline_integration_with_env():
    processor = DataPreprocessor(ticker="^NSEI", start_date="2023-01-01", end_date="2023-06-01")
    features, prices = processor.get_processed_data()

    env = IndiaMarketTradingEnv(features, prices)
    obs, info = env.reset()

    assert env.observation_space.contains(obs)
    assert obs.shape == (features.shape[1] + prices.shape[1],)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert not terminated
    assert step_info["portfolio_value"] > 0


def test_get_train_test_data_chronological_split():
    processor = DataPreprocessor(ticker="^NSEI", start_date="2023-01-01", end_date="2024-06-01")
    (train_features, train_prices), (test_features, test_prices) = processor.get_train_test_data(split_date="2024-01-01")

    assert not train_features.empty
    assert not test_features.empty
    assert len(train_features) == len(train_prices)
    assert len(test_features) == len(test_prices)

    # Chronological integrity: all train dates < 2024-01-01 and all test dates >= 2024-01-01
    assert (train_features.index < "2024-01-01").all()
    assert (test_features.index >= "2024-01-01").all()

    # Scaling integrity: Train features mean should be close to 0 and std close to 1
    assert np.allclose(train_features.mean().values, 0.0, atol=1e-1)
    assert np.allclose(train_features.std().values, 1.0, atol=1e-1)


if __name__ == "__main__":
    test_data_pipeline()
    test_data_pipeline_integration_with_env()
    test_get_train_test_data_chronological_split()
