# Mô hình GRU – đối chứng cho MNN

Thư mục này chứa mô hình **GRU** dùng để so sánh với mạng nơ-ron ma trận (MNN) ở thư mục gốc. Toàn bộ luồng dữ liệu, huấn luyện và đánh giá được giữ nguyên như MNN – **chỉ thay đổi mô hình**. Mã nguồn gốc không bị chỉnh sửa.

## Yêu cầu

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib cartopy
```

Đã kiểm tra với TensorFlow 2.19 / Keras 3 (khác với MNN dùng `tf.compat.v1`).

## Cách chạy

Từ thư mục gốc của project:

```bash
python GRU/GRU_Main.py
```

Trên Windows, nếu console báo lỗi `UnicodeEncodeError` khi in tiếng Việt:

```powershell
$env:PYTHONIOENCODING = "utf-8"; python GRU/GRU_Main.py
```

Dữ liệu `ibtracs-tbd.csv` được đọc từ thư mục gốc, có thể chạy lệnh từ bất kỳ thư mục nào.

> **Thời gian:** khoảng 90–115 giây/epoch trên CPU → 200 epoch mất khoảng **5–6 giờ**.

## Chỉnh kích thước mô hình

Sửa `gru_cfg` ở đầu [GRU_Main.py](GRU_Main.py):

```python
gru_cfg = dict(
    gru_units=(64, 64),   # số unit của từng lớp GRU xếp chồng
    dense_units=32,        # lớp Dense ẩn trước đầu ra (0/None để bỏ)
    dropout=0.1,
    use_layernorm=True,
)
```

| Tham số | Ý nghĩa |
|---|---|
| `gru_units` | Tuple số unit mỗi lớp GRU; số phần tử = số lớp (vd. `(32,)`, `(64, 64)`, `(128, 64, 32)`) |
| `dense_units` | Số nơ-ron lớp Dense ẩn (hàm `tanh`); `0` hoặc `None` để nối thẳng ra đầu ra |
| `dropout` | Tỉ lệ dropout sau mỗi lớp GRU; `0` để tắt (MNN không dùng dropout) |
| `use_layernorm` | Bật/tắt LayerNorm sau mỗi lớp GRU (MNN có dùng LayerNorm) |

Xem mục [Đưa về cùng quy mô với MNN](#đưa-về-cùng-quy-mô-với-mnn) để chọn cấu hình có số tham số tương đương MNN.

Xem nhanh cấu trúc và số tham số:

```bash
python GRU/GRU_Model.py
```

## Đưa về cùng quy mô với MNN

Để so sánh công bằng, nên chọn cấu hình sao cho số tham số xấp xỉ mạng MNN. MNN gốc (`first_hid_cfg = [2, 25, 8]`, `second_hid_cfg = [2, 50, 10]`) có **20,522 tham số**; cấu hình mặc định của GRU trong thư mục này có **40,994**.

### Cấu hình gợi ý (≈ 20,522 tham số)

Giữ `dense_units=32`, `use_layernorm=True`, chỉ đổi `gru_units`:

| Phương án | `gru_units` | Số tham số | Chênh lệch so với MNN |
|---|---|---|---|
| **2 lớp** (khuyến nghị – tương ứng 2 lớp ẩn của MNN) | `(44, 44)` | 20,294 | -1.1% |
| 1 lớp | `(74,)` | 20,596 | +0.4% |

Ví dụ trong [GRU_Main.py](GRU_Main.py):

```python
gru_cfg = dict(
    gru_units=(44, 44),
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

### Công thức số tham số của GRU

Gọi `d` là kích thước đầu vào của một lớp (lớp đầu `d = 5`), `u` là số unit của lớp đó.

```
P_cell(d, u) = 3 · u · (d + u + 2)
```

(3 cổng; Keras mặc định `reset_after=True` nên có 2 vector bias cho mỗi cổng.)

- Đầu ra mỗi lớp có kích thước `u` → đó là `d` của lớp kế tiếp
- LayerNorm sau mỗi lớp: `2 × (kích thước đầu ra của lớp)`
- Dense ẩn: `(u_cuối + 1) · dense_units`; Dense đầu ra: `(dense_units + 1) · 2` (nếu `dense_units=0` thì là `(u_cuối + 1) · 2`)
- Dropout không có tham số

**Dạng rút gọn** khi các lớp có cùng số unit `u`, `dense_units=32`, `use_layernorm=True`:

| Số lớp | Tổng số tham số | Hệ số `(a, b)` |
|---|---|---|
| 1 lớp | `3·u² + 55·u + 98` | `(3, 55)` |
| 2 lớp | `9·u² + 63·u + 98` | `(9, 63)` |

### Tự tính lại khi đổi quy mô MNN

Giải phương trình `a·u² + b·u + 98 = P_MNN` để tìm `u`:

```python
import math

a1, a2, b1, b2, w = 25, 8, 50, 10, 4          # cấu hình MNN mới
P_MNN = a1*a2*(w + 8) + b1*b2*(a1 + a2 + 3) + 2*(b1 + b2 + 1)

a, b = 9, 63                               # hệ số GRU 2 lớp (bảng trên)
u = round((-b + math.sqrt(b*b - 4*a*(98 - P_MNN))) / (2*a))
print(P_MNN, u)
```

Kiểm tra lại bằng số tham số thực tế của Keras (chạy trong thư mục `GRU/`):

```bash
python -c "from GRU_Model import build_gru_model as f; print(f(gru_units=(44, 44)).count_params())"
```

## Kiến trúc

```
Đầu vào (4 × 5)
→ GRU → LayerNorm → Dropout      (lặp theo gru_units)
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
- Cách chia nhóm cho học hai giai đoạn (hàm `split_two_stage_vars` trong [GRU_Training.py](GRU_Training.py)):

  | MNN | GRU |
  |---|---|
  | Nhóm R: `W_R` (nhân theo chiều đặc trưng) | `kernel` đầu vào của các lớp GRU (x_t · W) |
  | Nhóm L: `W_L`, bias, LayerNorm | `recurrent_kernel`, bias, LayerNorm, các lớp Dense |

## Tệp mã nguồn

| Tệp | Chức năng |
|---|---|
| `GRU_Main.py` | Chương trình chính (sao chép từ `MNN_Main.py`, chỉ đổi mô hình và nơi lưu kết quả) |
| `GRU_Model.py` | Hàm `build_gru_model` tạo cấu trúc mô hình |
| `GRU_Training.py` | Huấn luyện hai giai đoạn, lưu mô hình tốt nhất |
| `GRU_Testing.py` | Nạp mô hình tốt nhất và dự báo trên tập kiểm thử |

## Kết quả đầu ra

Tất cả được lưu trong thư mục `GRU/`, không ghi đè kết quả của MNN:

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
