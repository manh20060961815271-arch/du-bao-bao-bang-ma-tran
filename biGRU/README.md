# Mô hình biGRU – đối chứng cho MNN

Thư mục này chứa mô hình **GRU hai chiều (Bidirectional GRU)** dùng để so sánh với mạng nơ-ron ma trận (MNN) ở thư mục gốc. Toàn bộ luồng dữ liệu, huấn luyện và đánh giá được giữ nguyên như MNN – **chỉ thay đổi mô hình**. Mã nguồn gốc không bị chỉnh sửa.

Cấu trúc thư mục và cách chạy giống hệt thư mục [GRU](../GRU/README.md).

## Yêu cầu

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib cartopy
```

Đã kiểm tra với TensorFlow 2.19 / Keras 3.

## Cách chạy

Từ thư mục gốc của project:

```bash
python biGRU/biGRU_Main.py
```

Trên Windows, nếu console báo lỗi `UnicodeEncodeError` khi in tiếng Việt:

```powershell
$env:PYTHONIOENCODING = "utf-8"; python biGRU/biGRU_Main.py
```

> **Thời gian:** khoảng 90 giây/epoch trên CPU → 200 epoch mất khoảng **5 giờ**.

## Chỉnh kích thước mô hình

Sửa `bigru_cfg` ở đầu [biGRU_Main.py](biGRU_Main.py):

```python
bigru_cfg = dict(
    gru_units=(64, 64),   # số unit của từng lớp biGRU xếp chồng (mỗi chiều)
    dense_units=32,        # lớp Dense ẩn trước đầu ra (0/None để bỏ)
    dropout=0.1,
    use_layernorm=True,
    merge_mode="concat",   # cách ghép 2 chiều: "concat", "sum", "mul", "ave"
)
```

| Tham số | Ý nghĩa |
|---|---|
| `gru_units` | Tuple số unit **mỗi chiều** của từng lớp biGRU; số phần tử = số lớp |
| `dense_units` | Số nơ-ron lớp Dense ẩn (hàm `tanh`); `0` hoặc `None` để nối thẳng ra đầu ra |
| `dropout` | Tỉ lệ dropout sau mỗi lớp biGRU; `0` để tắt (MNN không dùng dropout) |
| `use_layernorm` | Bật/tắt LayerNorm sau mỗi lớp biGRU |
| `merge_mode` | Cách ghép đầu ra hai chiều; `"concat"` làm kích thước đầu ra gấp đôi `gru_units` |

Xem mục [Đưa về cùng quy mô với MNN](#đưa-về-cùng-quy-mô-với-mnn) để chọn cấu hình có số tham số tương đương MNN.

Xem nhanh cấu trúc và số tham số:

```bash
python biGRU/biGRU_Model.py
```

## Đưa về cùng quy mô với MNN

Để so sánh công bằng, nên chọn cấu hình sao cho số tham số xấp xỉ mạng MNN. MNN gốc (`first_hid_cfg = [2, 25, 8]`, `second_hid_cfg = [2, 50, 10]`) có **20,522 tham số**; cấu hình mặc định của biGRU trong thư mục này có **106,466**.

### Cấu hình gợi ý (≈ 20,522 tham số)

Giữ `dense_units=32`, `use_layernorm=True`, `merge_mode="concat"`, chỉ đổi `gru_units`:

| Phương án | `gru_units` | Số tham số | Chênh lệch so với MNN |
|---|---|---|---|
| **2 lớp** (khuyến nghị – tương ứng 2 lớp ẩn của MNN) | `(27, 27)` | 20,996 | +2.3% |
| 1 lớp | `(50,)` | 20,598 | +0.4% |

Ví dụ trong [biGRU_Main.py](biGRU_Main.py):

```python
bigru_cfg = dict(
    gru_units=(27, 27),
    dense_units=32,
    dropout=0.1,
    use_layernorm=True,
    merge_mode="concat",
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

### Công thức số tham số của biGRU

Gọi `d` là kích thước đầu vào của một lớp (lớp đầu `d = 5`), `u` là số unit của lớp đó.

```
P_cell(d, u) = 3 · u · (d + u + 2)
```

(3 cổng; Keras mặc định `reset_after=True` nên có 2 vector bias cho mỗi cổng.)

- Mỗi lớp biGRU gồm **2** cell độc lập (xuôi + ngược): `P_lớp = 2 · P_cell(d, u)`
- Với `merge_mode="concat"`, đầu ra mỗi lớp có kích thước `2u` (với `"sum"`, `"mul"`, `"ave"` là `u`) → đó là `d` của lớp kế tiếp
- LayerNorm sau mỗi lớp: `2 × (kích thước đầu ra của lớp)`
- Dense ẩn: `(2·u_cuối + 1) · dense_units`; Dense đầu ra: `(dense_units + 1) · 2` (nếu `dense_units=0` thì là `(2·u_cuối + 1) · 2`)
- Dropout không có tham số

**Dạng rút gọn** khi các lớp có cùng số unit `u`, `dense_units=32`, `use_layernorm=True`, `merge_mode="concat"`:

| Số lớp | Tổng số tham số | Hệ số `(a, b)` |
|---|---|---|
| 1 lớp | `6·u² + 110·u + 98` | `(6, 110)` |
| 2 lớp | `24·u² + 126·u + 98` | `(24, 126)` |

### Tự tính lại khi đổi quy mô MNN

Giải phương trình `a·u² + b·u + 98 = P_MNN` để tìm `u`:

```python
import math

a1, a2, b1, b2, w = 25, 8, 50, 10, 4          # cấu hình MNN mới
P_MNN = a1*a2*(w + 8) + b1*b2*(a1 + a2 + 3) + 2*(b1 + b2 + 1)

a, b = 24, 126                               # hệ số biGRU 2 lớp (bảng trên)
u = round((-b + math.sqrt(b*b - 4*a*(98 - P_MNN))) / (2*a))
print(P_MNN, u)
```

Kiểm tra lại bằng số tham số thực tế của Keras (chạy trong thư mục `biGRU/`):

```bash
python -c "from biGRU_Model import build_bigru_model as f; print(f(gru_units=(27, 27)).count_params())"
```

## Kiến trúc

```
Đầu vào (4 × 5)
→ Bidirectional(GRU) → LayerNorm → Dropout      (lặp theo gru_units)
→ Dense (tanh) → Dense(2) → Reshape
Đầu ra (1 × 2)
```

Chiều xuôi đọc cửa sổ theo thứ tự thời gian (t-3 → t), chiều ngược đọc từ t về t-3; trạng thái ẩn của hai chiều được ghép theo `merge_mode`.

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
- Cách chia nhóm cho học hai giai đoạn (hàm `split_two_stage_vars` trong [biGRU_Training.py](biGRU_Training.py)), giống thư mục GRU:

  | MNN | biGRU |
  |---|---|
  | Nhóm R: `W_R` (nhân theo chiều đặc trưng) | `kernel` đầu vào của GRU chiều xuôi **và** chiều ngược |
  | Nhóm L: `W_L`, bias, LayerNorm | `recurrent_kernel`, bias của cả hai chiều, LayerNorm, các lớp Dense |

## Tệp mã nguồn

| Tệp | Chức năng |
|---|---|
| `biGRU_Main.py` | Chương trình chính (sao chép từ `MNN_Main.py`, chỉ đổi mô hình và nơi lưu kết quả) |
| `biGRU_Model.py` | Hàm `build_bigru_model` tạo cấu trúc mô hình |
| `biGRU_Training.py` | Huấn luyện hai giai đoạn, lưu mô hình tốt nhất |
| `biGRU_Testing.py` | Nạp mô hình tốt nhất và dự báo trên tập kiểm thử |

## Kết quả đầu ra

Tất cả được lưu trong thư mục `biGRU/`, không ghi đè kết quả của MNN hay GRU:

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
