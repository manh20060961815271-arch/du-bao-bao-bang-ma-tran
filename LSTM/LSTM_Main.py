import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import MultipleLocator

import LSTM_Testing as Testing
import LSTM_Training as Training

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.metrics import r2_score

def haversine_distance(coords_true, coords_pred):
    R = 6371.0  
    lon1, lat1 = np.radians(coords_true[:, 0]), np.radians(coords_true[:, 1])
    lon2, lat2 = np.radians(coords_pred[:, 0]), np.radians(coords_pred[:, 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2.0) ** 2 + 
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c

def load_csv_data(csv_path, window_size=4):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file '{csv_path}'!")

    df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
    
    df['STORM_SPEED'] = pd.to_numeric(df['STORM_SPEED'], errors='coerce')
    df['STORM_DIR'] = pd.to_numeric(df['STORM_DIR'], errors='coerce')
    
    cols = ['STORM_SPEED', 'STORM_DIR']
    df[cols] = df.groupby('ID', sort=False)[cols].ffill().fillna(0.0)
    
    df = df.reset_index(drop=True)
    
    cyclones_data = []
    initial_coords = [] 

    for storm_id, group in df.groupby('ID', sort=False):
        
        lon = group['LON'].to_numpy(dtype=np.float64)
        lat = group['LAT'].to_numpy(dtype=np.float64)
        speed = group['STORM_SPEED'].to_numpy(dtype=np.float64)
        
        dir_rad = np.radians(group['STORM_DIR'].to_numpy(dtype=np.float64))
        sin_dir = np.sin(dir_rad)
        cos_dir = np.cos(dir_rad)
        
        coords = np.column_stack((lon, lat, speed, sin_dir, cos_dir))
        start_lon, start_lat = coords[0, 0], coords[0, 1] 

        # Xử lý bẫy vượt kinh tuyến 180 độ 
        lon_diff = coords[:, 0] - start_lon
        lon_diff = (lon_diff + 180) % 360 - 180

        coords[:, 0] = lon_diff
        coords[:, 1] = coords[:, 1] - start_lat

        if coords.ndim == 2 and coords.shape[0] >= 8:
            cyclones_data.append(coords)
            initial_coords.append((start_lon, start_lat))

    return cyclones_data, initial_coords

# 1. NẠP DỮ LIỆU
WINDOW_SIZE = 4

# Cấu hình mô hình LSTM (chỉnh kích thước tại đây)
lstm_cfg = dict(
    lstm_units=(64, 64),   # số unit của từng lớp LSTM xếp chồng
    dense_units=32,        # lớp Dense ẩn trước đầu ra (0/None để bỏ)
    dropout=0.1,
    use_layernorm=True,
)

print('1. Đang nạp dữ liệu bão (Tích hợp Tọa độ, Vận tốc và Hướng di chuyển)...')
all_cyclones, all_initials = load_csv_data(os.path.join(ROOT_DIR, 'ibtracs-tbd.csv'), window_size=WINDOW_SIZE)

train_size = int(len(all_cyclones) * 0.7)
val_size = int(len(all_cyclones) * 0.8)

train_final = all_cyclones[:train_size]
val_final = all_cyclones[train_size:val_size]
test_final = all_cyclones[val_size:]
test_initials = all_initials[val_size:]

# 2. HUẤN LUYỆN
print(f'\n2. Quá trình Huấn luyện (Train: {len(train_final)}, Val: {len(val_final)})...')
scaler_X, scaler_Y, train_losses, val_losses = Training.training(
    data_train=train_final, data_val=val_final, window_size=WINDOW_SIZE,
    model_cfg=lstm_cfg
)

print('-> Đang vẽ đồ thị hàm Loss...')
plt.figure(figsize=(10, 5))
plt.plot(train_losses, color='blue', linewidth=1.5, label='Training Loss')
plt.plot(val_losses, color='orange', linewidth=1.5, label='Validation Loss')
plt.title('Biểu đồ hàm Loss qua các Epochs (LSTM)')
plt.xlabel('Epoch')
plt.ylabel('Giá trị Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig(os.path.join(BASE_DIR, 'Training_Loss_Curve.png'), dpi=150, bbox_inches='tight')
plt.close()

# 3. KIỂM THỬ VÀ KHÔI PHỤC TỌA ĐỘ GỐC
print('\n3. Quá trình Kiểm thử...')
horizon = 4
_, _, Y_true_disp, Y_pred_disp, valid_indices = Testing.testing(
    data_test=test_final, window_size=WINDOW_SIZE,
    model_cfg=lstm_cfg,
    scaler_X=scaler_X, scaler_Y=scaler_Y
)

Y_true_abs = []
Y_pred_abs = []

for i, idx in enumerate(valid_indices):
    start_lon, start_lat = test_initials[idx]
    storm_data = test_final[idx]
    
    row_num = storm_data.shape[0]
    
    storm_true_abs = []
    storm_pred_abs = []
    
    for sample_idx, j in enumerate(range(WINDOW_SIZE - 1, row_num - horizon)):
        # Tọa độ tương đối của điểm neo so với vị trí khởi đầu của bão
        anchor_rel_lon, anchor_rel_lat = storm_data[j, 0], storm_data[j, 1]
        
        # Tọa độ tuyệt đối của điểm neo
        anchor_abs_lon = anchor_rel_lon + start_lon
        anchor_abs_lat = anchor_rel_lat + start_lat
        
        # Cộng độ dời vào điểm neo tuyệt đối
        true_lon = anchor_abs_lon + Y_true_disp[i][sample_idx, 0]
        true_lat = anchor_abs_lat + Y_true_disp[i][sample_idx, 1]
        
        pred_lon = anchor_abs_lon + Y_pred_disp[i][sample_idx, 0]
        pred_lat = anchor_abs_lat + Y_pred_disp[i][sample_idx, 1]
        
        storm_true_abs.append([true_lon, true_lat])
        storm_pred_abs.append([pred_lon, pred_lat])
        
    Y_true_abs.append(np.array(storm_true_abs))
    Y_pred_abs.append(np.array(storm_pred_abs))

a_test_abs = np.concatenate(Y_pred_abs, axis=0)
b_test_abs = np.concatenate(Y_true_abs, axis=0)


# 4. ĐÁNH GIÁ (TRÊN TỌA ĐỘ TUYỆT ĐỐI)
mse_score = mean_squared_error(b_test_abs, a_test_abs)
all_point_distances = haversine_distance(b_test_abs, a_test_abs)
mean_km_error = np.mean(all_point_distances)
rmse_km_error = np.sqrt(np.mean(all_point_distances**2)) 

print(f'\nMSE: {mse_score:.5f} | MAE (Haversine): {mean_km_error:.2f} km | RMSE (Haversine): {rmse_km_error:.2f} km')


print('\n--- BẢNG THỐNG KÊ PHÂN BỐ SAI SỐ ---')

error_bins = {
    '< 40 km': 0, '40 - 70 km': 0, '70 - 90 km': 0, 
    '90 - 120 km': 0, '120 - 150 km': 0, '> 150 km': 0
}

for i, idx in enumerate(valid_indices):
    true_traj = Y_true_abs[i].reshape(-1, 2)
    pred_traj = Y_pred_abs[i].reshape(-1, 2)
    
    if len(true_traj) == 0 or len(pred_traj) == 0: 
        continue
    
    storm_err = np.mean(haversine_distance(true_traj, pred_traj))
    
    if storm_err < 40: error_bins['< 40 km'] += 1
    elif storm_err < 70: error_bins['40 - 70 km'] += 1
    elif storm_err < 90: error_bins['70 - 90 km'] += 1
    elif storm_err < 120: error_bins['90 - 120 km'] += 1
    elif storm_err < 150: error_bins['120 - 150 km'] += 1
    else: error_bins['> 150 km'] += 1

df_stats = pd.DataFrame(list(error_bins.items()), columns=['Khoảng sai số', 'Số lượng cơn bão'])
print(df_stats.to_string(index=False))

plt.figure(figsize=(10, 5))
labels = list(error_bins.keys())
counts = list(error_bins.values())

bars = plt.bar(labels, counts, color='orange', edgecolor='blue', width=0.6, zorder=3)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), 
             ha='center', va='bottom', fontweight='bold', color='black')

plt.title('Thống kê Số lượng Cơn bão theo Khoảng Sai số', fontsize=12)
plt.xlabel('Khoảng sai số trung bình (km)', fontsize=11)
plt.ylabel('Số lượng cơn bão', fontsize=11)
plt.grid(True, axis='y', linestyle='--', alpha=0.6, zorder=0)

output_chart = os.path.join(BASE_DIR, 'Error_Distribution_Chart.png')
plt.savefig(output_chart, dpi=150, bbox_inches='tight')
plt.close()
print(f"-> Đã xuất đồ thị thống kê sai số ra tệp '{output_chart}'")

# THÊM VÀO PHẦN 4: VẼ ĐỒ THỊ R2 PHÂN TÁN CHO KINH ĐỘ VÀ VĨ ĐỘ
from sklearn.metrics import r2_score

# Tách riêng dữ liệu Kinh độ (cột 0) và Vĩ độ (cột 1) từ toạ độ tuyệt đối
obs_lon = b_test_abs[:, 0]
obs_lat = b_test_abs[:, 1]
pred_lon = a_test_abs[:, 0]
pred_lat = a_test_abs[:, 1]

# Tạo khung đồ thị 1x2 
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

def plot_scatter(ax, obs, pred, title_prefix):
    r2 = r2_score(obs, pred)
    ax.scatter(obs, pred, s=10, c="#3110A7", label=f'R2 = {r2:.3f}')
    
    # Tính đường tham chiếu y = x
    min_val = min(np.min(obs), np.min(pred))
    max_val = max(np.max(obs), np.max(pred))
    ax.plot([min_val, max_val], [min_val, max_val], color='#0072B2', linestyle='-')
    
    ax.set_xlabel(f'Observed {title_prefix} value')
    ax.set_ylabel(f'Prediction {title_prefix} value')
    ax.set_title(f'Prediction vs Observed: {title_prefix}')
    ax.legend(loc='upper left', frameon=True)

# Vẽ đồ thị Vĩ độ 
plot_scatter(axs[0], obs_lat, pred_lat, 'latitude')
# Vẽ đồ thị Kinh độ 
plot_scatter(axs[1], obs_lon, pred_lon, 'longitude')

plt.tight_layout()
output_r2_chart = os.path.join(BASE_DIR, 'R2_Scatter_Plot.png')
plt.savefig(output_r2_chart, dpi=150, bbox_inches='tight')
plt.close()
print(f"-> Đã xuất đồ thị phân tán R2 ra tệp '{output_r2_chart}'")
# ----------------------------------------------------------------------

# 5. VẼ BẢN ĐỒ QUỸ ĐẠO 
FORECAST_HOURS = 12 

def _wrap180(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0

def plot_track_map(track, true_pts, pred_pts, title, out_path, pad=5.0, min_span=20.0):
    lon_all = np.concatenate([track[:, 0], pred_pts[:, 0]])
    lat_all = np.concatenate([track[:, 1], pred_pts[:, 1]])
    c = np.degrees(np.arctan2(np.mean(np.sin(np.radians(lon_all))),
                              np.mean(np.cos(np.radians(lon_all)))))
    x = lambda lon: _wrap180(lon - c)

    x_all = x(lon_all)
    x0, x1 = x_all.min() - pad, x_all.max() + pad
    y0, y1 = lat_all.min() - pad, lat_all.max() + pad
    if x1 - x0 < min_span:
        m = (x0 + x1) / 2; x0, x1 = m - min_span / 2, m + min_span / 2
    if y1 - y0 < min_span * 0.75:
        m = (y0 + y1) / 2; y0, y1 = m - min_span * 0.375, m + min_span * 0.375
    y0, y1 = max(y0, -85), min(y1, 85)

    proj = ccrs.PlateCarree(central_longitude=c)
    fig, ax = plt.subplots(figsize=(8, 6.5), subplot_kw={'projection': proj})
    ax.set_extent([x0, x1, y0, y1], crs=proj)
    
    ax.add_feature(cfeature.OCEAN, facecolor='#a9d3e9')
    ax.add_feature(cfeature.LAND, facecolor='#f6f0d6')
    ax.coastlines(resolution='50m', linewidth=0.6)
    
    gl = ax.gridlines(draw_labels=True, linestyle=':', linewidth=0.6, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    kw = {'transform': proj}

    # Quỹ đạo thực (xanh) và dự báo (cam)
    ax.plot(x(track[:, 0]), track[:, 1], '-o', color='#1f77b4', ms=4, lw=1.5,
            label='Quỹ đạo thực tế', zorder=3, **kw)
    ax.plot(x(pred_pts[:, 0]), pred_pts[:, 1], '-o', color='#ff7f0e', ms=4, lw=1.2,
            label=f'Dự báo +{FORECAST_HOURS}h', zorder=4, **kw)

    d = haversine_distance(true_pts, pred_pts)
    ax.text(0.02, 0.98, f'N={len(d)}\nMAE={d.mean():.1f} km\nRMSE={np.sqrt((d ** 2).mean()):.1f} km',
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), zorder=5)

    y_bar = y0 + 0.07 * (y1 - y0)
    km_per_deg = 111.32 * np.cos(np.radians(y_bar))
    L_km = max([k for k in (100, 200, 500, 1000) if k / km_per_deg <= 0.25 * (x1 - x0)] or [100])
    L = L_km / km_per_deg
    xb1 = x1 - 0.05 * (x1 - x0); xb0 = xb1 - L
    ax.plot([xb0, xb1], [y_bar, y_bar], 'k-', lw=3, zorder=5, **kw)
    ax.text((xb0 + xb1) / 2, y_bar + 0.02 * (y1 - y0), f'{L_km} km', ha='center', va='bottom',
            fontsize=9, fontweight='bold', zorder=5, **kw)

    ax.set_title(title)
    ax.legend(loc='lower left', fontsize='small')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


#  VÒNG LẶP XUẤT TOÀN BỘ CƠN BÃO
output_dir = os.path.join(BASE_DIR, 'Cyclone_Maps_All')
os.makedirs(output_dir, exist_ok=True)
count = 0

for i, idx in enumerate(valid_indices):
    true_traj = Y_true_abs[i].reshape(-1, 2)
    pred_traj = Y_pred_abs[i].reshape(-1, 2)
    
    if len(true_traj) == 0 or len(pred_traj) == 0: 
        continue

    start_lon, start_lat = test_initials[idx]
    track = test_final[idx][:, :2] + np.array([start_lon, start_lat])
    
    storm_avg_error = np.mean(haversine_distance(true_traj, pred_traj))
    out_path = os.path.join(output_dir, f'Map_{idx}_MAE{storm_avg_error:.0f}km.png')
    
    plot_track_map(track, true_traj, pred_traj,
                   title=f'DỰ BÁO QUỸ ĐẠO BÃO (+{FORECAST_HOURS}H) - LSTM - ID: {idx}',
                   out_path=out_path)
    count += 1

print(f'-> Đã xuất toàn bộ {count} bản đồ quỹ đạo vào thư mục "{output_dir}/".')