# BÁO CÁO TOÀN DIỆN PIPELINE SOTA: UWB FALL DETECTION
## 9th ABC Challenge — UWB Fall Detection Track
**Tác giả**: Lead AI Architect & PyTorch Team  
**Mô hình SOTA**: Bi-Path Dynamic Biomechanical Triple-Ensemble  
**Kết quả chính thức**: **Validation Ranking F1 = 0.9077** (F1@0.3 = 0.9231 | FA/h = 1.46)  
**Tệp nộp bài Test**: `submission_dynamic_ensemble.csv` (101 sự kiện, 100% Validated)  
**Notebook thực thi**: `ABC2027_Dynamic_Triple_Ensemble.ipynb`

---

## 1. TỔNG QUAN BÀI TOÁN & THIẾT KẾ KIẾN TRÚC TOÀN DIỆN

### 1.1. Bối cảnh Vật lý & Bản chất Tín hiệu Radar UWB
- **Môi trường thu thập**: Căn hộ thông minh **LIARA** (Đại học UQAC, Canada) diện tích khoảng $40\text{ m}^2$.
- **Hệ thống cảm biến**: 3 radar xung UWB **Novelda XeThru X4M200** với tần số phát xung trung tâm $7.29\text{ GHz}$, đặt ở 3 góc nhìn cố định trong căn hộ.
- **Định dạng dữ liệu thô**: Mỗi luồng liên tục $30\text{ phút}$ được lưu trong tệp `.h5` với tensor Channel Impulse Response (CIR):
  $$\mathbf{R} \in \mathbb{R}^{3 \times T \times 368}$$
  - $3$ radar độc lập (`radar_serials = [29171, 29444, 30722]`).
  - $T$ khung thời gian chậm (*slow-time frames*) được đồng bộ hóa ở tần số $f_s = 50.0\text{ Hz}$ ($\Delta t = 0.02\text{ s}$). Một luồng $30\text{ phút}$ có $T \approx 88,500\text{ frames}$.
  - $368$ ô cự ly thời gian nhanh (*fast-time range bins*), độ phân giải vật lý $\sim 5.3\text{ cm}$ mỗi bin (phủ cự ly từ $0\text{ m}$ đến gần $20\text{ m}$).
- **Cơ chế ghép nối luồng (*Stream Concatenation*)**: Ban tổ chức ghép ngẫu nhiên $\sim 140$ phân đoạn ADL (15 hoạt động thường nhật), ngã (4 kịch bản `Chute01`–`Chute04`), và phòng trống (`Empty`/`Autre`) thành luồng liên tục. Giữa các đoạn có vùng trộn tuyến tính **crossfade $0.2\text{ s}$** ($10\text{ frames}$).

### 1.2. Thước đo Đánh giá Chính thức của Cuộc thi
Theo bộ đánh giá chuẩn của ban tổ chức ([`abc2027.evaluate`](file:///D:/Desktop/ABC27/abc2027_uwb_train_v1/src/abc2027/evaluate.py)):
- Dự đoán hợp lệ gồm 4 cột: `[stream_id, t_start, t_end, score]` (và cột tuỳ chọn `t_alarm`).
- Cơ chế ghép cặp tham lam 1-1 theo điểm số `score` giảm dần tại các ngưỡng Temporal Intersection-over-Union $\theta \in \{0.1, 0.3, 0.5, 0.7\}$.
- Vùng miễn trừ (`--ignore-flagged`): Các ca ngã chất lượng thấp (`quality = low` tại nệm 3) hoặc cần xem xét (`needs_review = true`) được mở rộng lề $\pm 2.0\text{ s}$ thành **vùng ignore** — mô hình chạm vùng này không được cộng TP và không bị phạt FP.
- **Điểm xếp hạng tổng thể (Ranking Score)**:
  $$\text{Ranking F1} = \frac{F_1(\theta=0.1) + F_1(\theta=0.3) + F_1(\theta=0.5)}{3}$$
- **Tiêu chí phụ phá hòa (Tie-Breaks)**:
  1. *Tie-Break 1*: Sai số tuyệt đối trung vị của thời điểm bắt đầu tại $\theta=0.3$ (`start_error_median_abs`, càng thấp càng tốt).
  2. *Tie-Break 2*: Số lượng báo động giả trên mỗi giờ tại $\theta=0.3$ (`false_alarms_per_hour`, càng thấp càng tốt).

### 1.3. Sơ đồ Kiến trúc Pipeline Hoàn chỉnh (End-to-End Architecture)

```
[Luồng UWB CIR thô: 3 Radars x T x 368 Bins @ 50Hz]
   │
   ├───> [NHÁNH 1: HỌC SÂU (Deep Learning Path)]
   │        │
   │        ├── 1. Khử Clutter: Causal EMA (tau = 3.0s, alpha = 0.99336, lfilter_zi)
   │        ├── 2. Cắt cự ly không gian: Bins [0 : 100] (vùng hoạt động 0 - 5.3m)
   │        ├── 3. Chuẩn hóa Stream-level Z-Score theo từng radar: (Amp - mean) / std
   │        │
   │        └── 4. Suy luận Test-Time Augmentation (TTA) bước mịn (Fine-Stride Hop = 5 frames = 0.10s)
   │                 ├── Model 1: ResNet-18 Sequence (w1 = 0.45) ───> p1(t) (Nén bin dim=3, Temporal Head)
   │                 ├── Model 2: ResNet-18 Standard (w2 = 0.40) ───> p2(t) (Global GAP dim=(2,3), FC)
   │                 └── Model 3: EfficientNet-B0 Seq (w3 = 0.15) ──> p3(t) (MBConv, Inverted Residuals)
   │                 │
   │                 └── Tích lũy 20x Overlap: P_ens(t) = 0.45*p1(t) + 0.40*p2(t) + 0.15*p3(t)
   │
   └───> [NHÁNH 2: ĐỘNG HỌC KHÔNG TRỄ PHA (Physical Kinematics Path)]
            │
            ├── 1. Khử Clutter cục bộ: Overlapped Gram-Matrix SVD (K = 2)
            │      Khối 8.0s (400 frames), Hop 4.0s (200 frames), Cửa sổ Hann w(t)
            │      G = X^T * X  ==>  X_clean = X - X * V_K * V_K^T
            │
            └── 2. Đường bao động năng không trễ pha E(t) & Đạo hàm bậc nhất dE/dt
                   E(t) = sum_r ||X_{clean, r}(t)||_2  ==>  Gaussian smoothing (sigma = 0.14s)
   │
   └───> [GIAI ĐOẠN TINH CHỈNH BIÊN SINH CƠ HỌC (Dynamic Biomechanical Refinement)]
            │
            ├── 1. Vùng đề xuất sơ bộ: P_ens(t) > 0.60, nối khoảng hở < 0.45s, min_width >= 1.20s, P_peak >= 0.675
            ├── 2. Lọc nhiễu ghép đoạn: Kiểm tra FWHM động năng > 0.68s (loại bỏ xung crossfade 0.2s)
            ├── 3. Dò sườn bắt đầu t_start: Quét ngược từ t_peak tìm xung dE/dt > 0.20*max_g đầu tiên
            ├── 4. Dò đuôi kết thúc t_end: Quét xuôi từ va chạm đến khi E(t) <= base + 0.60*span (cắt bỏ nằm bất động)
            └── 5. Biomechanical Guard: Ràng buộc 1.40s <= (t_end - t_start) <= 3.50s (failsafe t_peak +/- 1.2s)
   │
   v
[Tệp Nộp Bài Chuẩn Hóa: submission_dynamic_ensemble.csv (101 sự kiện, PASS 100%)]
```

---

## 2. XỬ LÝ SỐ LIỆU & TIỀN XỬ LÝ TÍN HIỆU (SIGNAL PREPROCESSING)

### 2.1. Nghịch lý Vật lý Giữa Khử Nhiễu Nhân Quả và Không Nhân Quả
Để mô hình đạt hiệu năng SOTA, hệ thống triển khai kiến trúc **Dual-Path Signal Conditioning** để giải quyết mâu thuẫn giữa 2 phương pháp khử nhiễu tĩnh:
- **Bộ lọc EMA ($\tau = 3.0\text{ s}$)**: Là bộ lọc nhân quả thời gian thực. Ưu điểm là tương thích 100% với không gian đặc trưng của các checkpoint ResNet/EfficientNet đã huấn luyện. Tuy nhiên, đáp ứng quá độ kéo dài $(1-\alpha)e^{-t/3.0}$ gây ra **độ trễ pha $0.6\text{s} - 1.2\text{s}$**, kéo đỉnh xác suất $t_{\text{peak}}$ dịch về sau và không thể xác định chính xác thời điểm va chạm.
- **Bộ lọc SVD toàn luồng (Whole-Stream SVD)**: Thất bại vì luồng $30\text{ phút}$ ghép từ 3 phòng/nệm khác nhau; $K=2$ thành phần kỳ dị toàn cục để lại sàn tĩnh DC khổng lồ ($0.035$) giữa các đoạn.
- **Giải pháp Đột phá — Overlapped Gram-Matrix SVD ($K = 2$) theo khối $8.0\text{ s}$**:
  Chia tín hiệu CIR thô thành các khối $400\text{ frames}$ ($8.0\text{ s}$), bước nhảy $200\text{ frames}$ ($4.0\text{ s}$) có trọng số cửa sổ Hann $w(t)$:
  Trên mỗi khối $\mathbf{X}_r \in \mathbb{R}^{L \times 368}$, xây dựng ma trận Gram không gian:
  $$\mathbf{G}_r = \mathbf{X}_r^\top \mathbf{X}_r \in \mathbb{R}^{368 \times 368}$$
  Phân tích trị riêng bằng `np.linalg.eigh`:
  $$\mathbf{G}_r \mathbf{v}_i = \lambda_i \mathbf{v}_i \quad (i = 1, \dots, 368)$$
  Lấy $K = 2$ vector riêng $\mathbf{V}_K = [\mathbf{v}_{367}, \mathbf{v}_{368}]$ ứng với 2 trị riêng lớn nhất. Tín hiệu chuyển động sạch:
  $$\mathbf{X}_{r, \text{clean}} = \mathbf{X}_r - (\mathbf{X}_r \mathbf{V}_K)\mathbf{V}_K^\top$$
  $\mathbf{v}_1$ hấp thụ hoàn toàn phản xạ tĩnh của tường và đồ đạc; $\mathbf{v}_2$ hấp thụ phản xạ cơ thể nằm bất động trên sàn. Kết quả tạo ra đường bao động năng $E(t)$ **hoàn toàn không trễ pha**.

### 2.2. Trích xuất Đặc trưng cho Nhánh Học Sâu (Deep Feature Branch)
1. **Lọc nền EMA**:
   $$c(t) = \alpha c(t-1) + (1-\alpha) r(t), \qquad y_{\text{clean}}(t) = r(t) - c(t)$$
   với $\alpha = e^{-\Delta t / 3.0} \approx 0.99336$, khởi tạo trạng thái xác lập qua `scipy.signal.lfilter_zi`.
2. **Cắt dải cự ly không gian (Spatial Range Bin Slicing)**:
   Chỉ lấy $100$ bin cự ly đầu: $\mathbf{A}_r(t, b) = |y_{\text{clean}, r}(t, b)|$ với $b \in [0, 99]$.
   - $100$ bin ứng với khoảng cách cự ly $0\text{ m} - 5.3\text{ m}$ (bao quát toàn bộ diện tích căn hộ LIARA).
   - Triệt tiêu hoàn toàn xuyên âm trực tiếp giữa ăng-ten Tx-Rx và các phản xạ đa đường từ trần/tường ở các bin xa ($101 - 367$).
3. **Chuẩn hóa Z-Score theo từng Radar (Stream-Level Per-Radar Z-Score)**:
   $$\tilde{\mathbf{A}}_r(t, b) = \frac{\mathbf{A}_r(t, b) - \mu_r}{\sigma_r + 10^{-6}}$$
   với $\mu_r, \sigma_r$ được tính trên toàn bộ luồng của radar $r$, đảm bảo bất biến với thang đo công suất phát của từng cảm biến.

---

## 3. KIẾN TRÚC MÔ HÌNH SOTA & CƠ CHẾ HUẤN LUYỆN (MODEL & TRAINING)

### 3.1. Thiết kế Bộ Ba Mô Hình (Triple-Ensemble Architecture)
Mỗi cửa sổ trượt $2.0\text{ s}$ ($100\text{ frames} \times 100\text{ bins} \times 3\text{ radars}$) được đưa đồng thời qua 3 mạng nơ-ron:

```
Window Tensor: (B, 3, 100, 100)
    │
    ├──> [ResNetSequenceClassifier] (w = 0.45)
    │       ResNet-18 Backbone -> Feature Map (B, 512, 4, 4)
    │       Pool Range Dim 3: .mean(dim=3) -> (B, 512, 4)
    │       1D Temporal Head: Linear(512, 128) -> ReLU -> Linear(128, 1) -> (B, 4, 1)
    │       Linear Interpolation -> (B, 100) -> Sigmoid -> p1(t)
    │
    ├──> [ResNetV1Standard] (w = 0.40)
    │       ResNet-18 Backbone -> Feature Map (B, 512, 4, 4)
    │       Global Spatio-Temporal GAP: .mean(dim=(2, 3)) -> (B, 512)
    │       Classifier Head: Linear(512, 128) -> ReLU -> Linear(128, 2)
    │       Softmax -> p2 in [0, 1] (gán cho toàn bộ 100 frames)
    │
    └──> [EfficientNetSequenceClassifier] (w = 0.15)
            EfficientNet-B0 Backbone (7 MBConv stages with SE) -> (B, 1280, 4, 4)
            Pool Range Dim 3: .mean(dim=3) -> (B, 1280, 4)
            1D Temporal Classifier: Linear(1280, 128) -> ReLU -> Linear(128, 1)
            Linear Interpolation -> (B, 100) -> Sigmoid -> p3(t)
```

### 3.2. Hàm Mất Mát Chuyên Biệt (Physics-Aware Composite Loss)
Được thiết kế để giải quyết triệt để 3 vấn đề: (1) Mất cân bằng dữ liệu $1:58$; (2) Nhiễu gán nhãn biên; (3) Điểm nối ghép đoạn crossfade $0.2\text{s}$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal}}^{\text{soft}} + \lambda_{\text{splice}} \cdot \mathcal{L}_{\text{splice}} + \lambda_{\text{TV}} \cdot \mathcal{L}_{\text{TV}}$$

1. **Soft-Target Focal Loss ($\alpha = 0.85, \gamma = 2.0$)**:
   Nhãn $y_{\text{smooth}}(t)$ được làm trơn bằng 1D Gaussian kernel ($\sigma = 0.147\text{s}$, mở rộng $\pm 0.44\text{s}$):
   $$\mathcal{L}_{\text{pos}}(t) = -\alpha \cdot y_{\text{smooth}}(t) \cdot (1 - p(t))^\gamma \cdot \log(p(t))$$
   $$\mathcal{L}_{\text{neg}}(t) = -(1 - \alpha) \cdot (1 - y_{\text{smooth}}(t)) \cdot p(t)^\gamma \cdot \log(1 - p(t))$$
   $$\mathcal{L}_{\text{Focal}}^{\text{soft}} = \frac{1}{\sum M_{\text{valid}}(t)} \sum_{t=1}^T \left[ \mathcal{L}_{\text{pos}}(t) + \mathcal{L}_{\text{neg}}(t) \right] \cdot M_{\text{valid}}(t)$$
   Mặt nạ $M_{\text{valid}}(t) = 0.0$ tại các vùng bị gắn cờ `quality = low` hoặc `needs_review = true` (tránh rò rỉ gradient từ nhãn kém chất lượng).
2. **Splicing Anti-Clutter Hard-Negative Penalty ($\lambda_{\text{splice}} = 5.0, \tau_{\text{thresh}} = 0.30$)**:
   Xây dựng mặt nạ điểm nối $M_{\text{splice}}(t) = 1.0$ trong khoảng $[t_{\text{splice}} - 0.1\text{s}, t_{\text{splice}} + 0.6\text{s}]$:
   $$\text{Gate}(t) = M_{\text{splice}}(t) \cdot \mathbb{I}(p(t) > 0.30)$$
   $$\mathcal{L}_{\text{splice}}(t) = \text{Gate}(t) \cdot \left[ -(1 - \alpha)\, p(t)^\gamma \log(1 - p(t)) + 0.15 \cdot \max(0, p(t) - 0.30)^2 \right]$$
   $$\mathcal{L}_{\text{splice}} = \frac{5.0}{\sum M_{\text{valid}}(t)} \sum_{t=1}^T \mathcal{L}_{\text{splice}}(t)$$
   Phạt gấp **$5\times$** bất kỳ kích hoạt nào $> 0.30$ tại điểm nối ghép đoạn.
3. **Total Variation Temporal Regularization ($\lambda_{\text{TV}} = 0.08$)**:
   $$\mathcal{L}_{\text{TV}} = \frac{0.08}{\sum M_{\text{pair}}(t)} \sum_{t=2}^T |p(t) - p(t-1)| \cdot M_{\text{pair}}(t)$$
   Ép đường cong xác suất phải biến thiên liên tục, không bị chẻ đôi cú ngã khi qua vùng khuất sóng.

---

## 4. SUY LUẬN & TINH CHỈNH BIÊN SINH CƠ HỌC (INFERENCE & REFINEMENT)

### 4.1. Suy Luận TTA Bước Mịn (Fine-Stride TTA, $\text{Hop} = 5$)
- Cửa sổ trượt kích thước $W = 100\text{ frames}$ ($2.0\text{ s}$) trượt với bước nhảy $\text{Hop} = 5\text{ frames}$ ($0.10\text{ s}$).
- Độ phủ chồng lặp đạt **$20\times$ voting overlap**:
  $$P_{\text{ens}}(t) = 0.45 \cdot \frac{\sum p_1(t)}{C(t)} + 0.40 \cdot \frac{\sum p_2(t)}{C(t)} + 0.15 \cdot \frac{\sum p_3(t)}{C(t)}$$
  với $C(t)$ là số lần cửa sổ bao phủ frame $t$ ($C(t) = 20$ ở phần thân luồng).

### 4.2. Thuật Toán Tinh Chỉnh Biên Sinh Cơ Học (Dynamic Biomechanical Refinement)
1. **Phân đoạn đề xuất sơ bộ**:
   - Lọc các frame có $P_{\text{ens}}(t) > 0.60$. Nối các khoảng hở nhỏ $< 0.45\text{ s}$ thành vùng ứng viên $[r_{\text{start}}, r_{\text{end}}]$.
   - Loại bỏ vùng có độ rộng $< 1.20\text{ s}$ hoặc điểm đỉnh $P_{\text{peak}} < 0.675$.
2. **Lọc nhiễu ghép nối đoạn qua FWHM Động năng SVD**:
   - Tính mức nền động năng cục bộ $\text{base} = P_{20}(\tilde{E})$ trong cửa sổ xung quanh $\pm 2.5\text{s}$.
   - Xác định đỉnh năng lượng $e_{\text{peak}} = \max \tilde{E}$ và độ vọt $\text{span} = e_{\text{peak}} - \text{base}$.
   - Tính độ rộng nửa biên độ $\text{FWHM}$: số khung hình có $\tilde{E}(t) > \text{base} + 0.50 \times \text{span}$.
   - **Quy tắc**: Nếu $\text{FWHM} < 0.68\text{ s}$ $\rightarrow$ Loại bỏ ngay lập tức (xóa sạch 80% xung kích ghép đoạn $0.2\text{s}$).
3. **Định vị thời điểm bắt đầu $t_{\text{start}}$ qua $\frac{dE}{dt}$**:
   - Quét ngược từ $t_{\text{peak}}$ về trước trong $[r_{\text{start}} - 0.65\text{s}, t_{\text{peak}}]$.
   - Tìm đỉnh đạo hàm dương đầu tiên thỏa mãn $\frac{d\tilde{E}}{dt} \ge 0.20 \times \max\left(\frac{d\tilde{E}}{dt}\right)$ và năng lượng $\tilde{E} \ge \text{base} + 0.16 \times \text{span}$.
   - Đi lùi đến chân dốc ($\frac{d\tilde{E}}{dt} \le 0.05 \times \max$) $\rightarrow$ Chốt chính xác thời điểm mất thăng bằng $t_{\text{start}}$ (triệt tiêu sai số trễ pha của EMA).
4. **Định vị thời điểm kết thúc $t_{\text{end}}$ qua mức ổn định năng lượng**:
   - Quét xuôi từ $e_{\text{peak}}$ đến khi động năng rơi xuống dưới ngưỡng ổn định:
     $$\tilde{E}(t) \le \text{base} + 0.60 \times \text{span}$$
   - Khung hình cuối cùng trước khi rơi xuống ngưỡng được chọn làm $t_{\text{end}}$ (cắt bỏ hoàn toàn giai đoạn nằm yên sau ngã).
5. **Rào chắn sinh cơ học (Biomechanical Failsafe)**:
   - Ràng buộc $1.40\text{ s} \le (t_{\text{end}} - t_{\text{start}}) \le 3.50\text{ s}$.
   - Nếu thời lượng $> 3.50\text{ s}$: Thu hẹp an toàn về khoảng $[t_{\text{peak}} - 1.2\text{s}, t_{\text{peak}} + 1.2\text{s}]$.

---

## 5. KẾT QUẢ THỰC NGHIỆM & ĐỐI SÁNH TOÀN DIỆN (RESULTS & BENCHMARK)

### 5.1. Đánh Giá Trên Tập Validation Chính Thức (`val_00` đến `val_04`, 2.05 Giờ)
Được chấm điểm trực tiếp bằng mã nguồn chính thức của ban tổ chức ([`abc2027.evaluate`](file:///D:/Desktop/ABC27/abc2027_uwb_train_v1/src/abc2027/evaluate.py) với `--ignore-flagged`):

| Thước đo (Official Metric) | Baseline (Cửa sổ cứng 2.0s) | Dynamic SVD Refinement v1 | **Dynamic Biomechanical Triple-Ensemble (SOTA)** |
| :--- | :---: | :---: | :---: |
| **Validation Ranking F1 (`f1_mean`)** | `0.7215` | `0.8424` | **`0.9077`** |
| **$F_1$ tại $\theta = 0.1$** | `0.7808` | `0.8992` | **`0.9385`** (`TP=61, FP=2, FN=6`) |
| **$F_1$ tại $\theta = 0.3$ (Primary Metric)** | `0.7671` | `0.8837` | **`0.9231`** (`TP=60, FP=3, FN=7`) |
| **$F_1$ tại $\theta = 0.5$ (High Overlap)** | `0.6164` | `0.7442` | **`0.8615`** (`TP=56, FP=7, FN=11`) |
| **Sai số $t_{\text{start}}$ trung vị (`start_error_median_abs`)** | `0.460 s` | `0.380 s` | **`0.320 s`** |
| **Tần suất báo động giả (`false_alarms_per_hour`)** | `11.23 FA/h` | `2.44 FA/h` | **`1.46 FA/h`** (Chỉ 3 FP trong 2.05h) |
| **Độ trùng khớp trung bình ($\text{Mean tIoU}$ tại $\theta=0.3$)** | `0.5821` | `0.6432` | **`0.6879`** |

### 5.2. Kết Quả Dự Đoán Trên Tập Test Chính Thức (`test_00` đến `test_04`, 2.07 Giờ)
Tệp nộp bài [`submission_dynamic_ensemble.csv`](file:///D:/Desktop/ABC27/submission_dynamic_ensemble.csv) đã được kiểm định bằng `validate_predictions`:
- **Tổng số sự kiện phát hiện**: **101 sự kiện**.
  - `test_00`: 21 sự kiện ($1769.80\text{ s}$).
  - `test_01`: 21 sự kiện ($1771.12\text{ s}$).
  - `test_02`: 23 sự kiện ($1697.12\text{ s}$).
  - `test_03`: 29 sự kiện ($1769.76\text{ s}$).
  - `test_04`: 7 sự kiện ($431.92\text{ s}$).
- **Đặc tính phân bố thời lượng**:
  - Thời lượng trung bình: **$2.27\text{ s}$** (trung vị $2.40\text{ s}$, độ lệch chuẩn $0.55\text{ s}$).
  - Thời lượng ngắn nhất: $1.40\text{ s}$ — Thời lượng dài nhất: $3.44\text{ s}$.
  - Trùng khớp tự nhiên với phân bố sinh cơ học của các kịch bản ngã trong bộ dữ liệu huấn luyện ($1.96\text{s} \pm 0.72\text{s}$).

### 5.3. So Sánh Với Tệp Submission Của Teammate (`submission.csv`)
- **Format**: Cả 2 bản đều đạt **PASS 100%** qua `validate_predictions` và đạt chuẩn 5 cột `stream_id, t_start, t_end, score, t_alarm`.
- **Độ đồng thuận**: 76 / 103 sự kiện trùng khớp ở $\text{tIoU} \ge 0.1$ ($73.8\%$), mean tIoU đạt **$0.6607$**.
- **Ưu thế vượt trội của bản SOTA**:
  1. *Triệt tiêu xung nhiễu ghép đoạn*: Bản teammate có các dự đoán rất ngắn bất thường như `test_00: [605.60, 606.58]` ($0.98\text{s}$) — bản SOTA đã loại bỏ triệt để nhờ bộ lọc FWHM động năng SVD.
  2. *Không bị cắt cụt cú ngã dài*: Bản SOTA bắt trọn vẹn các ca ngã 2 giai đoạn chới với kéo dài $\sim 3.0\text{s}$ (như `test_01: [362.02, 365.06]`), trong khi bản teammate bị giới hạn trần cứng ở $2.38\text{s}$.
  3. *Tối ưu tiêu chí phá hòa*: Bản SOTA đạt sai số $t_{\text{start}}$ chỉ **$0.320\text{s}$** và tỷ lệ báo động giả siêu thấp **$1.46\text{ FA/h}$**, tạo ưu thế quyết định khi tranh chấp thứ hạng cao trên Leaderboard.

---

## 6. HƯỚNG DẪN TÁI LẬP & CẤU TRÚC TỆP DỰ ÁN

Toàn bộ giải pháp được đóng gói độc lập và có thể chạy lại từ đầu bằng cách thực thi tuần tự các cell trong:
- **Notebook**: [`ABC2027_Dynamic_Triple_Ensemble.ipynb`](file:///D:/Desktop/ABC27/ABC2027_Dynamic_Triple_Ensemble.ipynb)
- **Tệp nộp bài Test**: [`submission_dynamic_ensemble.csv`](file:///D:/Desktop/ABC27/submission_dynamic_ensemble.csv)
- **Các trọng số mô hình đã huấn luyện trong thư mục `models/`**:
  - `models/resnet18_physics_composite.pth` ($w_1 = 0.45$)
  - `models/best_resnet18.pth` ($w_2 = 0.40$)
  - `models/efficientnet_b0_physics.pth` ($w_3 = 0.15$)
