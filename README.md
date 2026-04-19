# Banking Intent Unsloth Project

Dự án phân loại ý định ngân hàng sử dụng Llama-3-8B với thư viện Unsloth (tối ưu hóa huấn luyện PEFT/LoRA).

## Cấu trúc thư mục chuẩn
- `configs/`: Chứa cấu hình yaml.
- `scripts/`: Chứa mã nguồn thực thi.
- `sample_data/`: Nơi lưu trữ dữ liệu (train/test).
- `outputs/`: Nơi lưu model checkpoints sau khi huấn luyện.

## 1. Cài đặt môi trường
Sử dụng môi trường ảo Python 3.9+ và chạy lệnh sau để cài đặt các package cần thiết:
```bash
pip install -r requirements.txt
```

## 2. Quy trình chạy (Usage)

### Bước 1: Tiền xử lý dữ liệu (Preprocess)
Lấy dữ liệu từ HuggingFace `mteb/banking77`, lọc top 25 labels và chia split train/test:
```bash
python scripts/preprocess_data.py
```
> [!NOTE]
> File `train.csv` và `test.csv` sẽ được sinh tự động và lưu vào mục `sample_data/`.

### Bước 2: Huấn luyện mô hình (Train)
Huấn luyện SFT Llama-3 8B dùng thuật toán LoRA quantization 4-bit.
```bash
python scripts/train.py
```
Mô hình sau khi huấn luyện xong sẽ được lưu tại mục `outputs/`.

### Bước 3: Suy luận (Inference)
Kiểm thử mô hình bằng cách chạy text inference trực tiếp:
```bash
python scripts/inference.py
```

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
