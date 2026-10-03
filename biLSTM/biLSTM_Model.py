import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models

# Cùng định dạng dữ liệu với mạng MNN (xem Data_Processing.py):
#   Đầu vào X: (batch, window_size, 5) = [Δlon, Δlat, speed, sin(dir), cos(dir)] đã chuẩn hóa
#   Đầu ra  Y: (batch, 1, 2)           = độ dời (Δlon, Δlat) sau 12h đã chuẩn hóa
WINDOW_SIZE = 4
N_FEATURES = 5
N_OUTPUTS = 2


def build_bilstm_model(window_size=WINDOW_SIZE, n_features=N_FEATURES,
                       lstm_units=(64, 64), dense_units=32, dropout=0.1,
                       use_layernorm=True, merge_mode="concat", name="biLSTM_Cyclone"):
    inputs = layers.Input(shape=(window_size, n_features), name="input_window")
    x = inputs

    # Các lớp LSTM hai chiều xếp chồng: chiều xuôi đọc t-3 → t, chiều ngược đọc t → t-3,
    # lớp cuối chỉ trả về trạng thái ẩn cuối của mỗi chiều rồi ghép lại (merge_mode)
    for i, units in enumerate(lstm_units):
        is_last = (i == len(lstm_units) - 1)
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=not is_last),
            merge_mode=merge_mode, name=f"bilstm_{i + 1}")(x)
        if use_layernorm:
            x = layers.LayerNormalization(name=f"layernorm_{i + 1}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i + 1}")(x)

    # Khối hồi quy
    if dense_units:
        x = layers.Dense(dense_units, activation="tanh", name="dense_hidden")(x)
    x = layers.Dense(N_OUTPUTS, activation=None, name="dense_output")(x)

    # Đưa về đúng shape (batch, 1, 2) giống đầu ra MNN
    outputs = layers.Reshape((1, N_OUTPUTS), name="output_disp")(x)

    return models.Model(inputs=inputs, outputs=outputs, name=name)


if __name__ == "__main__":
    import numpy as np

    model = build_bilstm_model()
    model.summary()

    x_dummy = np.random.randn(8, WINDOW_SIZE, N_FEATURES).astype(np.float32)
    y_dummy = model.predict(x_dummy, verbose=0)
    print(f"Input shape: {x_dummy.shape} -> Output shape: {y_dummy.shape}")
