# Trained Model Checkpoints (`models/`)

Thư mục này lưu trữ 3 tệp trọng số PyTorch (`.pth`) của kiến trúc **Bi-Path Dynamic Biomechanical Triple-Ensemble** (được loại trừ khỏi Git qua `.gitignore` để giữ dung lượng repository gọn nhẹ):

| Tệp Checkpoint | Kiến trúc Mô hình | Trọng số Ensemble ($w_i$) | Đầu ra Suy luận |
| :--- | :--- | :---: | :--- |
| `resnet18_physics_composite.pth` | `ResNetSequenceClassifier` (ResNet-18 + Range Pool + 1D Temporal Head) | `0.45` | Chuỗi xác suất $p_1(t) \in [0, 1]^{100}$ |
| `best_resnet18.pth` | `ResNetV1Standard` (ResNet-18 + Global Spatio-Temporal GAP + FC) | `0.40` | Xác suất cửa sổ $p_2 \in [0, 1]$ |
| `efficientnet_b0_physics.pth` | `EfficientNetSequenceClassifier` (EfficientNet-B0 MBConv + SE + 1D Head) | `0.15` | Chuỗi xác suất $p_3(t) \in [0, 1]^{100}$ |

## Công thức Hợp nhất Xác suất (Fine-Stride TTA, $\text{Hop} = 5\text{ frames} = 0.10\text{s}$, $20\times$ Overlap)

$$P_{\text{ens}}(t) = 0.45 \cdot p_1(t) + 0.40 \cdot p_2(t) + 0.15 \cdot p_3(t)$$

Đặt cả 3 tệp `.pth` vào thư mục `models/` trước khi chạy `ABC2027_Dynamic_Triple_Ensemble.ipynb` hoặc `python -m src.inference.biomechanical_refinement`.
