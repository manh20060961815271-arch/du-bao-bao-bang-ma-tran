import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Data_Processing

import biGRU_Model
from biGRU_Training import SAVE_PATH


def testing(data_test, window_size, model_cfg, scaler_X, scaler_Y):
    tf.keras.backend.clear_session()

    model = biGRU_Model.build_bigru_model(window_size=window_size, n_features=5, **model_cfg)
    model.load_weights(SAVE_PATH)

    Y_pred, Y_true = [], []
    a_list, b_list = [], []

    valid_indices = []

    for idx, storm in enumerate(data_test):
        X, Y = Data_Processing.data_processing(storm, window_size)

        if len(X) == 0:
            continue

        valid_indices.append(idx)

        Y_true_tmp = np.array(Y, dtype=np.float64).reshape(-1, 2)
        Y_true.append(Y_true_tmp)

        X_scaled = scaler_X.transform(np.array(X, dtype=np.float64).reshape(-1, 5)).reshape(-1, window_size, 5)

        pred_scaled = model(X_scaled.astype(np.float32), training=False).numpy()

        storm_preds_np = np.array(pred_scaled).reshape(-1, 2)
        Y_pred_tmp = scaler_Y.inverse_transform(storm_preds_np)

        Y_pred.append(Y_pred_tmp)
        a_list.append(Y_pred_tmp)
        b_list.append(Y_true_tmp)

    a = np.concatenate(a_list, axis=0) if a_list else np.array([])
    b = np.concatenate(b_list, axis=0) if b_list else np.array([])

    return a, b, Y_true, Y_pred, valid_indices
