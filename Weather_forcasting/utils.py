def preprocess_data(df):
    # Handle missing values
    df.fillna(method='ffill', inplace=True)  # Example: Forward fill

    # Convert date/time if needed
    # df['date'] = pd.to_datetime(df['date'])

    # Feature engineering (if needed)
    # df['new_feature'] = df['feature1'] * df['feature2']

    return df