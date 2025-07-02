# 📁 File: app.py

import streamlit as st
import pandas as pd
import os
import joblib
from src.utils import load_config
from src.predict import forecast_future
from tensorflow.keras.models import load_model

st.set_page_config(page_title="🌊 Dự đoán mực nước 24h", layout="wide")
st.title("🌊 Dự đoán mực nước 24h bằng MTS-LSTM")

uploaded_file = st.file_uploader("📤 Tải lên file dữ liệu Excel theo giờ (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Lưu file tạm để dùng
    temp_path = "data/input_uploaded.xlsx"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Tải file thành công!")

    # Tải cấu hình
    config = load_config("config/camle_mtslstm.yaml")
    config["input_data"]["1h"] = temp_path
    config["input_data"]["1D"] = "data/Cam_Le_Daily.xlsx"  # giữ nguyên

    # Tải model đã huấn luyện
    model_path = "trained-models/mtslstm_model.keras"
    if not os.path.exists(model_path):
        st.error("❌ Không tìm thấy model đã huấn luyện!")
        st.stop()
    model = load_model(model_path)
    st.info("📦 Đã tải model huấn luyện sẵn.")

    # Load scalers
    scalers = {
        "1D": joblib.load("trained-models/scaler_1D.pkl"),
        "1h": joblib.load("trained-models/scaler_1h.pkl")
    }

    # Chạy dự đoán
    st.info("🤖 Đang chạy dự đoán...")
    pred_df = forecast_future(config, model, scalers, return_df=True)  # ⚠️ KHÔNG dùng scalers, return_df

    # Hiển thị kết quả
    st.success("✅ Dự đoán hoàn tất!")
    st.subheader("📈 Biểu đồ dự đoán mực nước")
    st.line_chart(pred_df.set_index("Datetime")["Muc_nuoc_du_doan"])

    st.subheader("📋 Bảng dữ liệu dự đoán")
    st.dataframe(pred_df)

    # Nút tải kết quả
    csv = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Tải kết quả CSV", data=csv, file_name="du_doan_muc_nuoc.csv", mime="text/csv")
