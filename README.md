
# FloodCast
FloodCast is a multivariate time-series forecasting model for urban flood prediction using MTS - LSTM networks.

```c
📁 mtslstm_project/
├── config/
│   └── camle_mtslstm.yaml         -> File YAML cấu hình
├── data/
│   ├── camle_daily.csv            -> Dữ liệu tần suất 1D
│   └── camle_hourly.csv           -> Dữ liệu tần suất 1h
├── src/
│   ├── preprocess.py              -> Xử lý dữ liệu (dựa vào config)
│   ├── model.py                   -> Xây dựng mô hình MTS-LSTM
│   ├── train.py                   -> Train model
│   ├── predict.py                 -> Dự đoán (gồm dự đoán 24 bước tiếp theo)
│   └── utils.py                   -> Tiện ích chung (đọc config, scaling,...)
├── outputs/
│   ├── predicted_water_level.png  -> Biểu đồ kết quả dự đoán
│   └── predicted_water_level.csv  -> Kết quả dự đoán
├── trained-models/
│   ├── mtslstm_model.keras        -> Mô hình huấn luyện
│   ├── scaler_1D.pkl             
│   └── scaler_1h.pkl
├── main.py                        -> Entry point (nạp config và gọi các bước)
├── app.py                         -> Local Web Testing
├── visualize_dem.py               -> Bản đồ số DEM
├── LSTM_Camle.ipynb               -> File Notebook follow LSTM model
└── README.md
```
