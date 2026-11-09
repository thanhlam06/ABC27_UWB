#!/bin/bash

echo "🚀 Bắt đầu đóng gói dự án UWB Fall Detection thành Git Repo..."

# 1. Khởi tạo Git repository
if [ ! -d ".git" ]; then
    git init
    echo "✅ Đã khởi tạo Git repository."
fi

# 2. Tạo tệp .gitignore (Bỏ qua dữ liệu thô và trọng số mô hình lớn)
cat <<EOT > .gitignore
__pycache__/
*.py[cod]
*\$py.class
venv/
.venv/
.env
.ipynb_checkpoints/
.pytest_cache/

# Bỏ qua data và models để tránh phình to Repo
/data/
/abc2027_uwb_train_v1/data/
*.h5
*.npy
*.npz
*.zip
models/*.pth
models/*.pt
checkpoints/*.pth
checkpoints/*.pt
uwb_tal_pipeline/outputs/*.pth
uwb_tal_pipeline/outputs/*.pt

# Bỏ qua các bản nháp notebook trùng lặp
*prompt hybrid.ipynb
*_kết quả.ipynb
EOT
echo "✅ Đã tạo tệp .gitignore."

# ==========================================
# GIAI ĐOẠN 1: CẤU HÌNH & KHỞI TẠO (SETUP)
# ==========================================
echo "⏳ Đang commit cấu hình dự án..."
touch README.md requirements.txt
git add .gitignore README.md requirements.txt
git commit -m "chore: Khởi tạo dự án, thiết lập .gitignore và cấu hình môi trường"

# ==========================================
# GIAI ĐOẠN 2: KHÁM PHÁ DỮ LIỆU (EDA)
# ==========================================
echo "⏳ Đang commit phần EDA..."
git add notebooks/*EDA*.ipynb src/data/ 2>/dev/null
git commit -m "docs/feat: Phân tích EDA phân bố thời lượng ngã và đặc tính tín hiệu UWB CIR"

# ==========================================
# GIAI ĐOẠN 3: TIỀN XỬ LÝ (PREPROCESSING)
# ==========================================
echo "⏳ Đang commit phần Preprocessing..."
git add src/preprocessing/ 2>/dev/null
git commit -m "feat: Xây dựng Dual-Path Signal Conditioning (Khử nhiễu Causal EMA & Overlapped SVD)"

# ==========================================
# GIAI ĐOẠN 4: HUẤN LUYỆN MÔ HÌNH (MODEL TRAINING)
# ==========================================
echo "⏳ Đang commit phần Model Training..."
git add src/models/ src/training/ 2>/dev/null
git commit -m "feat: Thiết kế Bi-Path Dynamic Triple-Ensemble (ResNet18, EfficientNet) và Physics-Aware Loss"

# ==========================================
# GIAI ĐOẠN 5: INFERENCE & KẾT QUẢ SOTA
# ==========================================
echo "⏳ Đang commit phần Kết quả và Tinh chỉnh..."
git add src/inference/ ABC2027_Dynamic_Triple_Ensemble.ipynb submission_dynamic_ensemble.csv 2>/dev/null
git add . # Gom tất cả các script phụ trợ còn sót lại
git commit -m "feat: Tích hợp Biomechanical Refinement và chốt kết quả Validation F1=0.9077"

# ==========================================
# CẤU HÌNH NHÁNH VÀ REMOTE
# ==========================================
git branch -M main

echo "🎉 Hoàn tất quá trình tạo lịch sử commit tuần tự!"
echo ""
echo "📌 Lịch sử commit của bạn hiện tại:"
git log --oneline --graph --decorate --all

echo ""
echo "Tiếp theo, hãy chạy lệnh sau để push code lên GitHub:"
echo "git remote add origin <URL_CỦA_BẠN>"
echo "git push -u origin main"
