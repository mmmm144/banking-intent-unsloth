import torch
import yaml
import os
import warnings
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore")

class IntentClassification:
    def __init__(self, model_path):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=128,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def __call__(self, message):
        prompt = f"Instruction: Classify the banking intent. \nInput: {message}\nOutput: "
        
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=32, 
                use_cache=True, 
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # Lấy phần intent được model generate
        prediction = decoded_output.split("Output: ")[-1].strip()
        # Xóa các dấu câu dư thừa ở cuối do model sinh lỗi
        prediction = prediction.strip("?.!")
        
        return prediction


if __name__ == "__main__":
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report

    # Trỏ đến dữ liệu test
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(base_dir, "sample_data", "test.csv")
    
    print(f"Đang nạp file dữ liệu test: {test_path}")
    test_df = pd.read_csv(test_path)
    
    print("Đang nạp Model từ logs...")
    model_path = os.path.join(base_dir, "outputs")
    clf = IntentClassification(model_path)

    print(f"Bắt đầu chấm điểm {len(test_df)} mẫu test... (quá trình này mất vài phút trên Kaggle)")
    y_true = test_df['label'].tolist()
    y_pred = []
    
    for i, text in enumerate(test_df['text']):
        pred = clf(text)
        y_pred.append(pred)
        if (i + 1) % 100 == 0:
            print(f"Đã xử lý {i + 1}/{len(test_df)} mẫu...")
        
    # Chấm điểm (Scoring)
    acc = accuracy_score(y_true, y_pred)
    

    
    print("\n" + "="*40)
    print("--- KẾT QUẢ ĐÁNH GIÁ (TEST RESULTS) ---")
    print("="*40)
    print(f"Độ chính xác (Accuracy): {acc * 100:.2f}%\n")
    print("Báo cáo phân loại chi tiết (Classification Report):")
    # Thêm zero_division=0 để tránh báo lỗi UndefinedMetricWarning
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Vẽ Confusion Matrix
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        print("Đang vẽ biểu đồ Confusion Matrix...")
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(24, 20))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix of Intent Classification', fontsize=20)
        plt.ylabel('True Intent', fontsize=16)
        plt.xlabel('Predicted Intent', fontsize=16)
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        plt.show()
    except ImportError:
        print("\n[MẸO] Bạn có thể cài thêm matplotlib và seaborn để xuất hình ảnh ma trận nhầm lẫn:")
        print("pip install matplotlib seaborn")