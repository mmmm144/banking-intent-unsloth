from unsloth import FastLanguageModel
import pandas as pd
import yaml
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

def load_config():
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "configs", "train.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def format_prompt(example):
    # Cần format theo dạng prompt để Llama-3 có thể hiểu
    return {
        "text": f"Instruction: Classify the banking intent. \nInput: {example['text']}\nOutput: {example['label']}<|eot_id|>"
    }

def main():
    config = load_config()

    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load data
    train_df = pd.read_csv(os.path.join(base_dir, "scripts", "sample_data", "train.csv"))
    test_df = pd.read_csv(os.path.join(base_dir, "scripts", "sample_data", "test.csv"))

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_dataset = train_dataset.map(format_prompt)
    test_dataset = test_dataset.map(format_prompt)

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        args=TrainingArguments(
            per_device_train_batch_size=config["batch_size"],
            learning_rate=float(config["learning_rate"]),
            num_train_epochs=config["epochs"],
            logging_steps=10,
            output_dir=os.path.join(base_dir, config["output_dir"]),
            save_strategy="epoch",
            eval_strategy="epoch",  # Cập nhật từ evaluation_strategy -> eval_strategy
        ),
    )

    trainer.train()

    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    print("Training complete!")

if __name__ == "__main__":
    main()