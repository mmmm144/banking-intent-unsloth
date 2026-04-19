import torch
import yaml
import os
from unsloth import FastLanguageModel

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
        
        return prediction


# TEST
if __name__ == "__main__":
    clf = IntentClassification("outputs")

    msg = "I lost my card"
    print("Input:", msg)
    print("Prediction:", clf(msg))