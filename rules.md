

## 🛠 Bộ quy tắc (Rules) cho Agent Code

### 1. Quy tắc về cấu trúc hệ thống (Structural Rules)
* [cite_start]**Tuân thủ sơ đồ thư mục:** Mọi file code được tạo ra phải nằm đúng trong cấu trúc: `scripts/`, `configs/`, hoặc `sample_data/`.
* [cite_start]**Độc lập và Module hóa:** Tách biệt hoàn toàn logic tiền xử lý (`preprocess_data.py`), huấn luyện (`train.py`) và suy luận (`inference.py`) [cite: 75-78].
* **Cấu hình tập trung:** Không được "hard-code" các thông số (như learning rate, path). [cite_start]Tất cả phải được đọc từ file `.yaml` trong thư mục `configs/`.

### 2. Quy tắc về cài đặt Inference (Mandatory Interface)
[cite_start]Đây là quy tắc quan trọng nhất để đảm bảo điểm số phần triển khai [cite: 55-63]:
* [cite_start]**Class Name:** Phải đặt tên lớp là `IntentClassification`[cite: 59].
* **Method `__init__`:** Chỉ nhận tham số `model_path`. [cite_start]Bên trong phải thực hiện load cấu hình, tokenizer và model checkpoint[cite: 56, 60].
* [cite_start]**Method `__call__`:** Chỉ nhận tham số `message` (string) và phải trả về `predicted_label` (string).

### 3. Quy tắc về thư viện Unsloth (Technical Rules)
* [cite_start]**Tối ưu hóa bộ nhớ:** Luôn sử dụng `FastLanguageModel` và thiết lập `load_in_4bit = True` để phù hợp với tài nguyên GPU [cite: 43-44].
* **LoRA Parameters:** Ưu tiên cấu hình `r=16`, `lora_alpha=32` và nhắm mục tiêu vào các module `q_proj, k_proj, v_proj, o_proj` để tối ưu cho tác vụ phân loại.
* **Inference Mode:** Trước khi dự đoán, phải gọi hàm `FastLanguageModel.for_inference(model)` để tăng tốc độ suy luận.

### 4. Quy tắc về Logging và Document (Reporting Rules)
* [cite_start]**README Standard:** Mỗi lần cập nhật code, phải cập nhật file `README.md` hướng dẫn cách cài đặt môi trường (`pip install`) và câu lệnh chạy script.
* [cite_start]**Hyperparameters Documentation:** Agent phải tự động liệt kê các thông số: Batch size, Learning rate, Optimizer, và Max sequence length vào phần báo cáo hoặc log.
