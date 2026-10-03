# Mô hình LSTM – đối chứng cho MNN

Thư mục này chứa mô hình **LSTM** dùng để so sánh với mạng nơ-ron ma trận (MNN) ở thư mục gốc. Toàn bộ luồng dữ liệu, huấn luyện và đánh giá được giữ nguyên như MNN – **chỉ thay đổi mô hình**. Mã nguồn gốc không bị chỉnh sửa.

## Yêu cầu

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib cartopy
```

Đã kiểm tra với TensorFlow 2.19 / Keras 3 (khác với MNN dùng `tf.compat.v1`).

## Cách chạy

Từ thư mục gốc của project:

```bash
python LSTM/LSTM_Main.py
```

Trên Windows, nếu console báo lỗi `UnicodeEncodeError` khi in tiếng Việt:

```powershell
$env:PYTHONIOENCODING = "utf-8"; python LSTM/LSTM_Main.py
```

Dữ liệu `ibtracs-tbd.csv` được đọc từ thư mục gốc, có thể chạy lệnh từ bất kỳ thư mục nào.

> **Thời gian:** khoảng 100 giây/epoch trên CPU → 200 epoch mất khoảng **6 giờ**.

## Chỉnh kích thước mô hình

Sửa `lstm_cfg` ở đầu [LSTM_Main.py](LSTM_Main.py):

```python
lstm_cfg = dict(
    lstm_units=(64, 64),   # số unit của từng lớp LSTM xếp chồng
    dense_units=32,        # lớp Dense ẩn trước đầu ra (0/None để bỏ)
    dropout=0.1,
    use_layernorm=True,
)
```

| Tham số | Ý nghĩa |
|---|---|
| `lstm_units` | Tuple số unit mỗi lớp LSTM; số phần tử = số lớp (vd. `(32,)`, `(64, 64)`, `(128, 64, 32)`) |
| `dense_units` | Số nơ-ron lớp Dense ẩn (hàm `tanh`); `0` hoặc `None` để nối thẳng ra đầu ra |
| `dropout` | Tỉ lệ dropout sau mỗi lớp LSTM; `0` để tắt (MNN không dùng dropout) |
| `use_layernorm` | Bật/tắt LayerNorm sau mỗi lớp LSTM (MNN có dùng LayerNorm) |

Xem mục [Đưa về cùng quy mô với MNN](#đưa-về-cùng-quy-mô-với-mnn) để chọn cấu hình có số tham số tương đương MNN.

Xem nhanh cấu trúc và số tham số:

```bash
python LSTM/LSTM_Model.py
```

## Đưa về cùng quy mô với MNN

Để so sánh công bằng, nên chọn cấu hình sao cho số tham số xấp xỉ mạng MNN. MNN gốc (`first_hid_cfg = [2, 25, 8]`, `second_hid_cfg = [2, 50, 10]`) có **20,522 tham số**; cấu hình mặc định của LSTM trong thư mục này có **53,346**.

### Cấu hình gợi ý (≈ 20,522 tham số)

Giữ `dense_units=32`, `use_layernorm=True`, chỉ đổi `lstm_units`:

| Phương án | `lstm_units` | Số tham số | Chênh lệch so với MNN |
|---|---|---|---|
| **2 lớp** (khuyến nghị – tương ứng 2 lớp ẩn của MNN) | `(39, 39)` | 20,846 | +1.6% |
| 1 lớp | `(65,)` | 20,768 | +1.2% |

Ví dụ trong [LSTM_Main.py](LSTM_Main.py):

```python
lstm_cfg = dict(
    lstm_units=(39, 39),
    dense_units=32,
    dropout=0.1,
    use_layernorm=True,
)
```

### Công thức số tham số của MNN

Một lớp ma trận biến đầu vào `n1 × n2` thành đầu ra `m1 × m2` có:

```
P_lớp = m1·m2·(n1 + n2 + 1)          (W_L: m1·m2·n1,  W_R: m1·m2·n2,  θ: m1·m2)
      + 2·m1·m2                       (γ, β của LayerNorm – chỉ ở 2 lớp ẩn)
```

Với `first_hid_cfg = [_, a1, a2]`, `second_hid_cfg = [_, b1, b2]`, `WINDOW_SIZE = w` và 5 đặc trưng đầu vào:

```
P_MNN = a1·a2·(w + 5 + 3)  +  b1·b2·(a1 + a2 + 3)  +  2·(b1 + b2 + 1)
```

Ví dụ cấu hình gốc: `200·12 + 500·36 + 2·61 = 2,400 + 18,000 + 122 = 20,522`.

### Công thức số tham số của LSTM

Gọi `d` là kích thước đầu vào của một lớp (lớp đầu `d = 5`), `u` là số unit của lớp đó.

```
P_cell(d, u) = 4 · u · (d + u + 1)
```

(4 cổng, mỗi cổng có kernel `d×u`, recurrent_kernel `u×u` và bias `u`.)

- Đầu ra mỗi lớp có kích thước `u` → đó là `d` của lớp kế tiếp
- LayerNorm sau mỗi lớp: `2 × (kích thước đầu ra của lớp)`
- Dense ẩn: `(u_cuối + 1) · dense_units`; Dense đầu ra: `(dense_units + 1) · 2` (nếu `dense_units=0` thì là `(u_cuối + 1) · 2`)
- Dropout không có tham số

**Dạng rút gọn** khi các lớp có cùng số unit `u`, `dense_units=32`, `use_layernorm=True`:

| Số lớp | Tổng số tham số | Hệ số `(a, b)` |
|---|---|---|
| 1 lớp | `4·u² + 58·u + 98` | `(4, 58)` |
| 2 lớp | `12·u² + 64·u + 98` | `(12, 64)` |

### Tự tính lại khi đổi quy mô MNN

Giải phương trình `a·u² + b·u + 98 = P_MNN` để tìm `u`:

```python
import math

a1, a2, b1, b2, w = 25, 8, 50, 10, 4          # cấu hình MNN mới
P_MNN = a1*a2*(w + 8) + b1*b2*(a1 + a2 + 3) + 2*(b1 + b2 + 1)

a, b = 12, 64                               # hệ số LSTM 2 lớp (bảng trên)
u = round((-b + math.sqrt(b*b - 4*a*(98 - P_MNN))) / (2*a))
print(P_MNN, u)
```

Kiểm tra lại bằng số tham số thực tế của Keras (chạy trong thư mục `LSTM/`):

```bash
python -c "from LSTM_Model import build_lstm_model as f; print(f(lstm_units=(39, 39)).count_params())"
```

## Kiến trúc

```
Đầu vào (4 × 5)
→ LSTM → LayerNorm → Dropout      (lặp theo lstm_units)
→ Dense (tanh) → Dense(2) → Reshape
Đầu ra (1 × 2)
```

Đầu vào/đầu ra giống hệt MNN: cửa sổ 4 thời điểm × 5 đặc trưng `[Δlon, Δlat, speed, sin(dir), cos(dir)]` → độ dời `(Δlon, Δlat)` sau 12 giờ (đã chuẩn hóa).

## Giống và khác so với MNN

**Giữ nguyên:**

- Nạp dữ liệu (`load_csv_data`), chia tập 70/10/20 theo thứ tự cơn bão, `WINDOW_SIZE = 4`
- Tạo cửa sổ trượt: import trực tiếp `Data_Processing.py` gốc
- `StandardScaler` fit trên tập huấn luyện
- Mỗi batch là một cơn bão, xáo thứ tự cơn bão mỗi epoch
- Loss MSE, Adam lr = 1e-4, cắt gradient [-1, 1], 200 epoch
- **Học hai giai đoạn luân phiên** với 2 bộ tối ưu Adam riêng (L rồi R)
- Lưu mô hình có loss kiểm định tốt nhất
- Toàn bộ phần đánh giá: MAE/RMSE Haversine, phân bố sai số, R², bản đồ quỹ đạo

**Khác:**

- Viết bằng TensorFlow 2 (`tf.GradientTape`, `@tf.function`) thay cho `tf.Session`.
- Cách chia nhóm cho học hai giai đoạn (hàm `split_two_stage_vars` trong [LSTM_Training.py](LSTM_Training.py)):

  | MNN | LSTM |
  |---|---|
  | Nhóm R: `W_R` (nhân theo chiều đặc trưng) | `kernel` đầu vào của các lớp LSTM (x_t · W) |
  | Nhóm L: `W_L`, bias, LayerNorm | `recurrent_kernel`, bias, LayerNorm, các lớp Dense |

## Tệp mã nguồn

| Tệp | Chức năng |
|---|---|
| `LSTM_Main.py` | Chương trình chính (sao chép từ `MNN_Main.py`, chỉ đổi mô hình và nơi lưu kết quả) |
| `LSTM_Model.py` | Hàm `build_lstm_model` tạo cấu trúc mô hình |
| `LSTM_Training.py` | Huấn luyện hai giai đoạn, lưu mô hình tốt nhất |
| `LSTM_Testing.py` | Nạp mô hình tốt nhất và dự báo trên tập kiểm thử |

## Kết quả đầu ra

Tất cả được lưu trong thư mục `LSTM/`, không ghi đè kết quả của MNN:

| Tệp / thư mục | Nội dung |
|---|---|
| `checkpoints/best.weights.h5` | Trọng số mô hình tốt nhất (bị xóa và tạo lại mỗi lần chạy) |
| `Training_Loss_Curve.png` | Loss huấn luyện và kiểm định theo epoch |
| `Error_Distribution_Chart.png` | Số lượng cơn bão theo khoảng sai số |
| `R2_Scatter_Plot.png` | Phân tán dự báo – quan trắc cho vĩ độ, kinh độ |
| `Cyclone_Maps_All/` | Bản đồ quỹ đạo từng cơn bão trong tập kiểm thử |

MSE, MAE và RMSE (km) được in ra màn hình ở cuối quá trình chạy.

## Lưu ý

Giống bản gốc, chương trình không cố định random seed nên kết quả giữa các lần chạy có thể khác nhau đôi chút.
