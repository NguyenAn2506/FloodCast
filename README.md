
# FloodCast
FloodCast is a multivariate time-series forecasting model for urban flood prediction using MTS - LSTM networks.

📁 mtslstm_project/
├── config/
│   └── camle_mtslstm.yaml       -> File YAML cấu hình
├── data/
│   ├── camle_daily.csv          -> Dữ liệu tần suất 1D
│   └── camle_hourly.csv         -> Dữ liệu tần suất 1h
├── src/
│   ├── preprocess.py            -> Xử lý dữ liệu (dựa vào config)
│   ├── model.py                 -> Xây dựng mô hình MTS-LSTM
│   ├── train.py                 -> Train model
│   ├── predict.py               -> Dự đoán (gồm dự đoán 24 bước tiếp theo)
│   └── utils.py                 -> Tiện ích chung (đọc config, scaling,...)
├── outputs/
│   ├── model_weights.h5         -> Model đã train
│   └── prediction_results.csv   -> Kết quả dự đoán
├── main.py                      -> Entry point (nạp config và gọi các bước)
├── app.py                       -> Local Web Testing
└── LSTM_Camle.ipynb             -> File Notebook follow LSTM model

