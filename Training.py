import os
import random
import shutil
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.preprocessing import StandardScaler

import Add_Layer
import Data_Processing

tf.disable_v2_behavior()
tf.compat.v1.enable_resource_variables() 

def algebraic_activation(x):
    return 1 * x / tf.sqrt(1.0 + tf.square(x))

def training(data_train, data_val, window_size, first_hid, second_hid):
    tf.reset_default_graph()
    BATCH_SIZE = None
    
    xs = tf.placeholder(tf.float32, [BATCH_SIZE, window_size, 5])
    ys = tf.placeholder(tf.float32, [BATCH_SIZE, 1, 2])

    (W_1, V_1, l1) = Add_Layer.add_layer(xs, [2, window_size, 5], first_hid, 
                                         activation_function=algebraic_activation, use_layernorm=True, layer_name="layer1")
    (W_2, V_2, l2) = Add_Layer.add_layer(l1, first_hid, second_hid, 
                                         activation_function=algebraic_activation, use_layernorm=True, layer_name="layer2")
    (W_3, V_3, prediction) = Add_Layer.add_layer(l2, second_hid, [BATCH_SIZE, 1, 2], 
                                                 activation_function=None, use_layernorm=False, layer_name="layer3")

    loss = tf.reduce_mean(tf.square(ys - prediction))
    learning_rate = 0.0001

   # 1. Kiến trúc Two-Stage Learning
    all_vars = tf.trainable_variables()
    
    # L: chứa W_L và các tham số phụ (theta, gamma, beta)
    vars_L = [v for v in all_vars if 'W_R' not in v.name]
    
    # R: CHỈ chứa W_R
    vars_R = [v for v in all_vars if 'W_R' in v.name]

    # Dùng 2 bộ tối ưu hóa riêng biệt
    optimizer_L = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer_R = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    grads_and_vars_L = optimizer_L.compute_gradients(loss, vars_L)
    clipped_L = [(tf.clip_by_value(g, -1.0, 1.0) if g is not None else g, v) for g, v in grads_and_vars_L]
    train_op_L = optimizer_L.apply_gradients(clipped_L)

    grads_and_vars_R = optimizer_R.compute_gradients(loss, vars_R)
    clipped_R = [(tf.clip_by_value(g, -1.0, 1.0) if g is not None else g, v) for g, v in grads_and_vars_R]
    train_op_R = optimizer_R.apply_gradients(clipped_R)

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
        train_data_processed.append((X_scaled, Y_scaled))

    val_data_processed = []
    for i in range(len(data_val)):
        [X, Y] = Data_Processing.data_processing(data_val[i], window_size)
        if len(X) > 0:
            X_scaled = scaler_X.transform(np.array(X).reshape(-1, 5)).reshape(-1, window_size, 5)
            Y_scaled = scaler_Y.transform(np.array(Y).reshape(-1, 2)).reshape(-1, 1, 2)
            val_data_processed.append((X_scaled, Y_scaled))

    save_dir = r'C:\tmp_cyclone_model'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model')

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    saver = tf.compat.v1.train.Saver(tf.trainable_variables())
    all_storm_losses, all_val_losses = [], []

    epochs = 200

    best_val_loss = float('inf')
    
    storm_indices = list(range(len(train_data_processed)))

    for epoch in range(epochs):
        epoch_losses = []
        random.shuffle(storm_indices)
        
        for i in storm_indices:
            X_scaled, Y_scaled = train_data_processed[i]
            
            # Thực thi 2 bước học luân phiên theo lý thuyết M-neuron
            sess.run(train_op_L, feed_dict={xs: X_scaled, ys: Y_scaled})
            _, current_loss = sess.run([train_op_R, loss], feed_dict={xs: X_scaled, ys: Y_scaled})
            
            epoch_losses.append(current_loss)
            
        avg_epoch_loss = np.mean(epoch_losses)
        all_storm_losses.append(avg_epoch_loss)

        val_epoch_losses = []
        for X_scaled, Y_scaled in val_data_processed:
            val_loss = sess.run(loss, feed_dict={xs: X_scaled, ys: Y_scaled})
            val_epoch_losses.append(val_loss)
            
        avg_val_loss = np.mean(val_epoch_losses)
        all_val_losses.append(avg_val_loss)
        
        print_msg = f'Epoch: {epoch + 1}/{epochs} | Train Loss: {avg_epoch_loss:.5f} | Val Loss: {avg_val_loss:.5f}'
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            saver.save(sess, save_path)
            print_msg += " -> Đã lưu Best Model!"
            
        print(print_msg)
        
    sess.close()
    
    return scaler_X, scaler_Y, all_storm_losses, all_val_losses