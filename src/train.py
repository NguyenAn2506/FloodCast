import os
import joblib 
from src.preprocess import prepare_datasets
from src.model import build_mtslstm_model
from src.utils import create_scalers
from tensorflow.keras.callbacks import EarlyStopping


def train_model(config):
    # Create scaler dict
    scalers = create_scalers()

    # Tạo dữ liệu train/test từ file + scaler
    X1D_train, X1h_train, y_train, X1D_test, X1h_test, y_test = prepare_datasets(config, scalers)

    input_shapes = {
        '1D': (X1D_train.shape[1], X1D_train.shape[2]),
        '1h': (X1h_train.shape[1], X1h_train.shape[2])
    }

    # Build model
    model = build_mtslstm_model(config, input_shapes)

    # Training
    print(model.summary())
    model.fit(
        [X1D_train, X1h_train], y_train,
        validation_data=([X1D_test, X1h_test], y_test),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    print("✅ Huấn luyện hoàn tất.")

     # 📁 Tạo thư mục nếu chưa tồn tại
    os.makedirs("trained-models", exist_ok=True)

    # 💾 Lưu model
    model.save("trained-models/mtslstm_model.keras")  # dùng định dạng mới
    print("💾 Đã lưu model tại: trained-models/mtslstm_model.keras")

    # 💾 Lưu scalers (thêm vào đây)
    joblib.dump(scalers['1D'], "trained-models/scaler_1D.pkl")
    joblib.dump(scalers['1h'], "trained-models/scaler_1h.pkl")
    print("💾 Đã lưu scaler vào thư mục trained-models")

    return model, scalers



