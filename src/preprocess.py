import numpy as np
from src.utils import load_excel_data, scale_features
from sklearn.model_selection import train_test_split


def create_sequences(data, seq_len, target_index, horizon=1):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len:i + seq_len + horizon, target_index])
    return np.array(X), np.array(y)


def prepare_datasets(config, scalers):
    # Load YAML fields
    daily_cfg = config['input_data']['daily_file']
    hourly_cfg = config['input_data']['hourly_file']
    features_1D = config['features']['1D']
    features_1h = config['features']['1h']
    target_col = config['target_variable']
    seq_1D = config['seq_length']['1D']
    seq_1h = config['seq_length']['1h']
    horizon = config['forecast_horizon']

    # Load data
    df_1d = load_excel_data(daily_cfg, 'Date')
    df_1h = load_excel_data(hourly_cfg, 'Datetime')

    # Ensure same time coverage
    df_1d = df_1d.dropna()
    df_1h = df_1h.dropna()

    # Scale inputs
    scaled_1D = scale_features(df_1d, features_1D, scalers['1D'])
    scaled_1h = scale_features(df_1h, features_1h, scalers['1h'])

    # Target column index
    target_idx_1h = df_1h.columns.get_loc(target_col)

    # Create sequences
    X_1D, _ = create_sequences(scaled_1D, seq_1D, target_index=0, horizon=1)  # No y here
    X_1h, y = create_sequences(scaled_1h, seq_1h, target_index=target_idx_1h, horizon=horizon)

    # Align lengths
    min_len = min(len(X_1D), len(X_1h), len(y))
    X_1D, X_1h, y = X_1D[-min_len:], X_1h[-min_len:], y[-min_len:]

    # Train/test split
    X1D_train, X1D_test, X1h_train, X1h_test, y_train, y_test = train_test_split(
        X_1D, X_1h, y, test_size=0.2, shuffle=False
    )

    return X1D_train, X1h_train, y_train, X1D_test, X1h_test, y_test
