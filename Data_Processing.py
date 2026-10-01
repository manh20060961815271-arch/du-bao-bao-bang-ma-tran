import numpy as np

def data_processing(table, window_size, forecast_steps=4):
    table = np.array(table, dtype=np.float64)

    if table.ndim == 3:
        table = np.squeeze(table, axis=1)

    row_num = table.shape[0]
    col_num = table.shape[1] if table.ndim > 1 else 1

    if col_num == 0:
        col_num = 2

    X = []
    Y = []

    if row_num <= window_size + forecast_steps - 1:
        return np.zeros((0, window_size, col_num), dtype=np.float64), np.zeros((0, 1, 2), dtype=np.float64)

    for j in range(window_size - 1, row_num - forecast_steps):
        
        # Cửa sổ đầu vào: 4 điểm kết thúc tại j
        window_tmp = table[j - window_size + 1 : j + 1, :].copy()
        
        # Lấy điểm j làm điểm neo 
        anchor_point = table[j, :2]
        
        window_tmp[:, :2] = window_tmp[:, :2] - anchor_point
        
        # du bao (4 bước = 12h)
        target_tmp = table[j + forecast_steps, :2]
        
        target_disp = (target_tmp - anchor_point).reshape(1, 2)
        
        X.append(window_tmp)
        Y.append(target_disp)

    if len(X) == 0:
        return np.zeros((0, window_size, col_num), dtype=np.float64), np.zeros((0, 1, 2), dtype=np.float64)

    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)