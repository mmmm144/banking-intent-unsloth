from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    dataset = load_dataset("mteb/banking77")
    df = pd.DataFrame(dataset["train"])

    # 🔥 chọn top 25 label (bạn có thể đổi 30)
    top_labels = df['label'].value_counts().head(25).index
    df = df[df['label'].isin(top_labels)]

    # normalize text
    df['text'] = df['text'].str.lower().str.strip()

    # overwrite label with the actual intent text (quan trọng)
    df['label'] = df['label_text']

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )

    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "sample_data")
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("Done preprocessing!")

if __name__ == "__main__":
    main()