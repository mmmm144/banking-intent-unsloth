import os
import argparse
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Đẩy mô hình lên Hugging Face Hub")
    parser.add_argument("--repo", type=str, required=True, help="Tên repo trên HuggingFace (VD: username/my-llama3-model)")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Write Token của bạn")
    
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "outputs")

    if not os.path.exists(output_dir):
        print(f"❌ Lỗi: Thư mục '{output_dir}' không tồn tại. Vui lòng train mô hình trước hoặc đặt đúng tên thư mục.")
        return

    print(f"🚀 Bắt đầu upload folder '{output_dir}' lên repository '{args.repo}' trên Hugging Face...")
    
    api = HfApi()
    
    # Tự động tạo Repostory (nếu chưa tồn tại)
    api.create_repo(repo_id=args.repo, repo_type="model", token=args.token, exist_ok=True)
    
    # Upload folder
    api.upload_folder(
        folder_path=output_dir,
        repo_id=args.repo,
        repo_type="model",
        token=args.token
    )
    
    print("✅ Upload hoàn tất! Giờ bạn có thể xóa thư mục outputs/ cho nhẹ máy.")

if __name__ == "__main__":
    main()
