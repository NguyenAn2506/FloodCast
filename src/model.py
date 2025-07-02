from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate


def build_mtslstm_model(config, input_shapes):
    seq_len_1D, num_feat_1D = input_shapes['1D']
    seq_len_1h, num_feat_1h = input_shapes['1h']
    forecast_horizon = config['forecast_horizon']

    # Input 1D branch
    input_1d = Input(shape=(seq_len_1D, num_feat_1D), name='input_1d')
    x1 = LSTM(64, return_sequences=False)(input_1d)

    # Input 1h branch
    input_1h = Input(shape=(seq_len_1h, num_feat_1h), name='input_1h')
    x2 = LSTM(128, return_sequences=False)(input_1h)

    # Concatenate branches
    x = Concatenate()([x1, x2])
    x = Dense(64, activation='relu')(x)
    output = Dense(forecast_horizon, name='output')(x)

    model = Model(inputs=[input_1d, input_1h], outputs=output)
    model.compile(optimizer='adam', loss='mse')

    return model
