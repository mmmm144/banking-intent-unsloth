# Banking Intent Unsloth Project

[![HuggingFace Model](https://img.shields.io/badge/🤗_HuggingFace-Model_Checkpoint-FFD21E.svg)](https://huggingface.co/aiMy144/banking-intent-unsloth)

Dự án phân loại ý định ngân hàng sử dụng Llama-3-8B với thư viện Unsloth (tối ưu hóa huấn luyện PEFT/LoRA).

## Cấu trúc thư mục chuẩn
- `configs/`: Chứa cấu hình yaml.
- `scripts/`: Chứa mã nguồn Python thực thi gốc.
- `sample_data/`: Nơi lưu trữ dữ liệu (train/test).
- `outputs/`: Nơi lưu model checkpoints sau khi huấn luyện.
- `train.sh`: Script tự động hóa toàn bộ quá trình Xử lý dữ liệu & Huấn luyện.
- `inference.sh`: Script tự động hóa quá trình Dự đoán & Đánh giá mô hình.

## 1. Cài đặt môi trường
Sử dụng môi trường ảo Python 3.9+ và chạy lệnh sau để cài đặt các package cần thiết:
```bash
pip install -r requirements.txt
```

## 2. Quy trình chạy (Usage)

Dự án được hệ thống hóa để chạy mượt mà thông qua các Bash script. Đảm bảo cấp quyền thực thi bằng lệnh `chmod +x train.sh inference.sh` trước khi chạy.

### Bước 1: Tiền xử lý dữ liệu & Huấn luyện (Train)
Bao gồm quy trình lấy dữ liệu từ HuggingFace `mteb/banking77`, lọc top 25 labels thành tập train/test. Sau đó tự động huấn luyện SFT Llama-3 8B dùng thuật toán LoRA quantization 4-bit.
```bash
./train.sh
# Hoặc trên Kaggle/Linux Server: bash train.sh
```
> [!NOTE]
> Tập dữ liệu CSV sẽ sinh tự động ở `sample_data/`. Mô hình sau khi huấn luyện sẽ lưu toàn bộ ở mục `outputs/`.
> *(Tùy chọn chạy thủ công bằng Python: `python scripts/preprocess_data.py` sau đó `python scripts/train.py`)*

### Bước 2: Suy luận & Cài đặt Đánh giá (Inference)
Chạy inference file để kiểm thử mô hình trực tiếp, tính toán Accuracy tổng và xuất Heatmap Confusion Matrix:
```bash
./inference.sh
# Hoặc trên Kaggle/Linux Server: bash inference.sh
```
> *(Tùy chọn chạy thủ công bằng Python: `python scripts/inference.py`)*

## 3. Siêu tham số cấu hình (Hyperparameters)
Hệ thống sử dụng các siêu tham số sau (cấu hình trong `configs/train.yaml`):

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| **Model** | `unsloth/llama-3-8b-Instruct-bnb-4bit` | Model 4-bit của Llama 3 tiết kiệm bộ nhớ GPU |
| **Max Seq Length** | `128` | Giới hạn số token đầu vào/ra |
| **Batch Size** | `8` | Số lượng mẫu học trong 1 batch |
| **Learning Rate** | `2e-4` | Tốc độ học |
| **Epochs** | `3` | Số vòng lặp huấn luyện toàn bộ dữ liệu |
| **LoRA R** | `16` | Hạng ma trận LoRA |
| **LoRA Alpha** | `32` | Scaling factor của LoRA |
| **LoRA Dropout** | `0.05` | Tỉ lệ dropout chống overfitting của LoRA |
| **Target Modules**| `["q_proj", "k_proj", "v_proj", "o_proj"]`| Áp dụng LoRA lên các trọng số attention cốt lõi |
