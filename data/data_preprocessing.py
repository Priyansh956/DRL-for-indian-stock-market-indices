import numpy as np
import pandas as pd
import yfinance as yf


class DataPreprocessor:
    """Handles data ingestion from Yahoo Finance and technical feature engineering

    for Indian market indices (NIFTY 50) and volatility (India VIX).
    """

    def __init__(
        self,
        ticker: str = "^NSEI",
        vix_ticker: str = "^INDIAVIX",
        start_date: str = "2018-01-01",
        end_date: str = "2026-01-01",
    ):
        self.ticker = ticker
        self.vix_ticker = vix_ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self) -> pd.DataFrame:
        """Downloads historical OHLCV data for index and volatility using yfinance."""
        print(f"📥 Downloading data for {self.ticker} and {self.vix_ticker}...")

        # Fetch Index data
        df = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
        )

        # Handle potential multi-index columns returned by newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Fetch India VIX data for market fear/volatility sentiment
        vix_df = yf.download(
            self.vix_ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
        )
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)

        df["VIX"] = vix_df["Adj Close"]

        # Drop missing rows resulting from holiday mismatches
        df = df.dropna()
        return df

    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes momentum, trend, and volatility indicators (RSI, MACD, ATR,

        rolling volatility).
        """
        data = df.copy()

        # 1. Log Returns
        data["Log_Return"] = np.log(data["Adj Close"] / data["Adj Close"].shift(1))

        # 2. Moving Averages (Trend)
        data["SMA_20"] = data["Adj Close"].rolling(window=20).mean()
        data["SMA_50"] = data["Adj Close"].rolling(window=50).mean()

        # 3. RSI (Relative Strength Index - 14 periods)
        delta = data["Adj Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        data["RSI"] = 100 - (100 / (1 + rs))

        # 4. MACD (Moving Average Convergence Divergence)
        exp12 = data["Adj Close"].ewm(span=12, adjust=False).mean()
        exp26 = data["Adj Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = exp12 - exp26
        data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

        # 5. ATR (Average True Range - 14 periods for volatility)
        high_low = data["High"] - data["Low"]
        high_close = np.abs(data["High"] - data["Adj Close"].shift())
        low_close = np.abs(data["Low"] - data["Adj Close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data["ATR"] = true_range.rolling(window=14).mean()

        # 6. Rolling Volatility (20-day annualized)
        data["Rolling_Vol"] = data["Log_Return"].rolling(window=20).std() * np.sqrt(250)

        # Drop rows with NaN values created by rolling windows
        data = data.dropna()
        return data

    def get_processed_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Runs the full ingestion and feature pipeline, returning features

        and raw price matrices ready for the DRL environment.
        """
        raw_df = self.fetch_data()
        feature_df = self.compute_technical_indicators(raw_df)

        # Separate feature matrix (scaled inputs) and price matrix (for portfolio valuation)
        feature_columns = [
            "RSI",
            "MACD",
            "MACD_Signal",
            "ATR",
            "Rolling_Vol",
            "VIX",
        ]

        # Standardize features using rolling z-scores or simple normalization to prevent scale explosions
        features = feature_df[feature_columns]
        features = (features - features.mean()) / (features.std() + 1e-8)

        # Price matrix containing actual Adj Close prices for execution
        prices = feature_df[["Adj Close"]]

        return features, prices
