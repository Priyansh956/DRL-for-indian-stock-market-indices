from data.data_preprocessing import DataPreprocessor


def test_data_pipeline():
    processor = DataPreprocessor(ticker="^NSEI", start_date="2023-01-01", end_date="2026-01-01")
    features, prices = processor.get_processed_data()

    # Assertions to ensure data integrity
    assert not features.empty, "Feature dataframe is empty"
    assert not prices.empty, "Price dataframe is empty"
    assert len(features) == len(prices), "Feature and price lengths must match"
    print("✅ Data Ingestion and Feature Engineering verified successfully!")
    print(f"Total processed rows: {len(features)}")


if __name__ == "__main__":
    test_data_pipeline()
