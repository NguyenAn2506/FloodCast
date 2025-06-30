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

    return model, scalers
