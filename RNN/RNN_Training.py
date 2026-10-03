import os
import sys
import random
import shutil
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Dùng lại hàm tạo cửa sổ trượt của bản gốc (thư mục cha)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Data_Processing

import RNN_Model

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
SAVE_PATH = os.path.join(SAVE_DIR, 'best.weights.h5')


def split_two_stage_vars(model):
    # Tương ứng với MNN:
    #   W_R (nhân theo chiều đặc trưng)  <->  kernel đầu vào của RNN (x_t · W)
    #   W_L + tham số phụ               <->  recurrent_kernel, bias, LayerNorm, Dense
    vars_R = [v for v in model.trainable_variables
              if v.path.endswith('rnn_cell/kernel')]
    ids_R = {id(v) for v in vars_R}
    vars_L = [v for v in model.trainable_variables if id(v) not in ids_R]
    return vars_L, vars_R


def training(data_train, data_val, window_size, model_cfg, epochs=200):
    tf.keras.backend.clear_session()

    model = RNN_Model.build_rnn_model(window_size=window_size, n_features=5, **model_cfg)

    learning_rate = 0.0001

    # 1. Kiến trúc Two-Stage Learning (giữ nguyên như MNN)
    vars_L, vars_R = split_two_stage_vars(model)

    # Dùng 2 bộ tối ưu hóa riêng biệt
    optimizer_L = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_R = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_L.build(vars_L)
    optimizer_R.build(vars_R)

    def compute_loss(ys, prediction):
        return tf.reduce_mean(tf.square(ys - prediction))

    spec = [tf.TensorSpec([None, window_size, 5], tf.float32),
            tf.TensorSpec([None, 1, 2], tf.float32)]

    @tf.function(input_signature=spec)
    def train_step_L(xs, ys):
        with tf.GradientTape() as tape:
            loss = compute_loss(ys, model(xs, training=True))
        grads = tape.gradient(loss, vars_L)
        clipped = [(tf.clip_by_value(g, -1.0, 1.0), v) for g, v in zip(grads, vars_L) if g is not None]
        optimizer_L.apply_gradients(clipped)
        return loss

    # Loss trả về được tính trước khi cập nhật R (giống sess.run([train_op_R, loss]))
    @tf.function(input_signature=spec)
    def train_step_R(xs, ys):
        with tf.GradientTape() as tape:
            loss = compute_loss(ys, model(xs, training=True))
        grads = tape.gradient(loss, vars_R)
        clipped = [(tf.clip_by_value(g, -1.0, 1.0), v) for g, v in zip(grads, vars_R) if g is not None]
        optimizer_R.apply_gradients(clipped)
        return loss

    @tf.function(input_signature=spec)
    def eval_step(xs, ys):
        return compute_loss(ys, model(xs, training=False))

    # 2. Chuẩn bị dữ liệu và Scaler
    scaler_X, scaler_Y = StandardScaler(), StandardScaler()
    all_X_train, all_Y_train = [], []
    train_raw_processed = []

    for i in range(len(data_train)):
        [X, Y] = Data_Processing.data_processing(data_train[i], window_size)
        if len(X) > 0:
            all_X_train.append(X)
            all_Y_train.append(Y)
            train_raw_processed.append((X, Y))

    all_X_train_np = np.concatenate(all_X_train, axis=0)
    all_Y_train_np = np.concatenate(all_Y_train, axis=0)
    scaler_X.fit(all_X_train_np.reshape(-1, 5))
    scaler_Y.fit(all_Y_train_np.reshape(-1, 2))

    train_data_processed = []
    for X, Y in train_raw_processed:
        X_scaled = scaler_X.transform(np.array(X).reshape(-1, 5)).reshape(-1, window_size, 5)
        Y_scaled = scaler_Y.transform(np.array(Y).reshape(-1, 2)).reshape(-1, 1, 2)
        train_data_processed.append((X_scaled.astype(np.float32), Y_scaled.astype(np.float32)))

    val_data_processed = []
    for i in range(len(data_val)):
        [X, Y] = Data_Processing.data_processing(data_val[i], window_size)
        if len(X) > 0:
            X_scaled = scaler_X.transform(np.array(X).reshape(-1, 5)).reshape(-1, window_size, 5)
            Y_scaled = scaler_Y.transform(np.array(Y).reshape(-1, 2)).reshape(-1, 1, 2)
            val_data_processed.append((X_scaled.astype(np.float32), Y_scaled.astype(np.float32)))

    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)

    all_storm_losses, all_val_losses = [], []

    best_val_loss = float('inf')

    storm_indices = list(range(len(train_data_processed)))

    for epoch in range(epochs):
        epoch_losses = []
        random.shuffle(storm_indices)

        for i in storm_indices:
            X_scaled, Y_scaled = train_data_processed[i]

            # Thực thi 2 bước học luân phiên giống MNN
            train_step_L(X_scaled, Y_scaled)
            current_loss = train_step_R(X_scaled, Y_scaled)

            epoch_losses.append(float(current_loss))

        avg_epoch_loss = np.mean(epoch_losses)
        all_storm_losses.append(avg_epoch_loss)

        val_epoch_losses = []
        for X_scaled, Y_scaled in val_data_processed:
            val_loss = eval_step(X_scaled, Y_scaled)
            val_epoch_losses.append(float(val_loss))

        avg_val_loss = np.mean(val_epoch_losses)
        all_val_losses.append(avg_val_loss)

        print_msg = f'Epoch: {epoch + 1}/{epochs} | Train Loss: {avg_epoch_loss:.5f} | Val Loss: {avg_val_loss:.5f}'

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_weights(SAVE_PATH)
            print_msg += " -> Đã lưu Best Model!"

        print(print_msg)

    return scaler_X, scaler_Y, all_storm_losses, all_val_losses
