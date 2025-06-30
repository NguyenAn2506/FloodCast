import yaml
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_excel_data(filepath, datetime_col):
    df = pd.read_excel(filepath)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(by=datetime_col)
    return df

def create_scalers():
    return {
        '1D': MinMaxScaler(),
        '1h': MinMaxScaler()
    }

def scale_features(df, features, scaler):
    scaled = scaler.fit_transform(df[features])
    return scaled

def inverse_scale(scaler, data):
    return scaler.inverse_transform(data)
