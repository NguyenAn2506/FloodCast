# 📁 File: app.py

import streamlit as st
import pandas as pd
import os
from src.train import train_model
from src.predict import forecast_future
from src.utils import load_config

st.set_page_config(page_title="MTS-LSTM Flood Forecast", layout="wide")

st.title("🌊 Dự đoán mực nước 24h bằng MTS-LSTM")

uploaded_file = st.file_uploader("📤 Tải lên file dữ liệu Excel (.xlsx) theo giờ", type=["xlsx"])

if uploaded_file is not None:
    # Lưu file tạm để mô hình sử dụng
    temp_input_path = "data/input_uploaded.xlsx"
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Tải file thành công!")

    # Tải cấu hình YAML
    config = load_config("config/camle_mtslstm.yaml")

    # Cập nhật đường dẫn dữ liệu vào cấu hình
    config["data"]["1h"] = temp_input_path  # dùng file vừa upload
    config["data"]["1D"] = "data/Cam_Le_Daily.xlsx"  # dữ liệu 1 ngày giữ nguyên

    st.info("🚀 Đang huấn luyện và dự đoán, vui lòng chờ trong giây lát...")

    model, scalers = train_model(config)
    pred_df = forecast_future(config, model, scalers, return_df=True)

    st.success("✅ Dự đoán hoàn tất!")

    # Hiển thị biểu đồ dự đoán
    st.subheader("📈 Biểu đồ mực nước dự đoán 24h")
    st.line_chart(data=pred_df.set_index("Datetime")["Muc_nuoc_du_doan"])

    # Hiển thị bảng dữ liệu
    st.subheader("📋 Dữ liệu dự đoán chi tiết")
    st.dataframe(pred_df)

    # Nút tải file kết quả
    csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Tải kết quả CSV", data=csv, file_name="du_doan_muc_nuoc.csv", mime='text/csv')


# Cài thư viện Streamlit 
## pip install streamlit

# Chạy Web App:  
## streamlit run app.py
### http://localhost:8501/
