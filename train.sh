#!/bin/bash

cd "$(dirname "$0")"

echo "====================================="
echo "BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN (TRAINING)"
echo "====================================="

if [ ! -f "sample_data/train.csv" ]; then
    echo "Phát hiện chưa xử lý dữ liệu. Đang chạy Preprocess Data..."
    python scripts/preprocess_data.py
fi

echo "Đang nạp mô hình và cấu hình để train..."
python scripts/train.py

echo "Hoàn tất Training! Model được lưu tại thư mục outputs/"
