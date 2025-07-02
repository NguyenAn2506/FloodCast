import numpy as np
import pandas as pd
from src.utils import load_excel_data, scale_features, inverse_scale
import matplotlib.pyplot as plt

def forecast_future(config, model, scalers, return_df=False):
    # Load config
    daily_file = config['input_data']['daily_file']
    hourly_file = config['input_data']['hourly_file']
    features_1D = config['features']['1D']
    features_1h = config['features']['1h']
    seq_1D = config['seq_length']['1D']
    seq_1h = config['seq_length']['1h']
    target_var = config['target_variable']
    horizon = config['forecast_horizon']

    # Load and process input
    df_1d = load_excel_data(daily_file, 'Date')
    df_1h = load_excel_data(hourly_file, 'Datetime')

    input_1D = df_1d[features_1D].dropna().values[-seq_1D:]
    input_1h = df_1h[features_1h].dropna().values[-seq_1h:]

    # Scale
    input_1D_scaled = scalers['1D'].transform(input_1D)
    input_1h_scaled = scalers['1h'].transform(input_1h)

    # Reshape
    X1D = np.expand_dims(input_1D_scaled, axis=0)
    X1h = np.expand_dims(input_1h_scaled, axis=0)

    # Predict
    y_pred_scaled = model.predict([X1D, X1h])

    # Inverse scale
    y_pred_unscaled = inverse_scale(scalers['1h'], np.column_stack([y_pred_scaled[0]] * len(features_1h)))
    y_pred_target = y_pred_unscaled[:, features_1h.index(target_var)]

    # Tạo thời gian
    last_time = df_1h['Datetime'].max()
    future_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=horizon, freq='H')

    # DataFrame kết quả
    pred_df = pd.DataFrame({
        'Datetime': future_times,
        'Muc_nuoc_du_doan': y_pred_target
    })

    # Lưu file
    pred_df.to_csv('outputs/predicted_water_level.csv', index=False)

    print("✅ Dự đoán hoàn tất! Kết quả lưu tại: outputs/predicted_water_level.csv")
    print(pred_df.head())

    # Hiển thị biểu đồ
    plt.figure(figsize=(10, 5))
    plt.plot(pred_df['Datetime'], pred_df['Muc_nuoc_du_doan'], marker='o', color='blue', label='Dự đoán 24h')
    plt.title("Dự đoán mực nước 24h tiếp theo")
    plt.xlabel("Thời gian")
    plt.ylabel("Mực nước (cm hoặc mm)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Trả về nếu được yêu cầu
    if return_df:
        return pred_df
