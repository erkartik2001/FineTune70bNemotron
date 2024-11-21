import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os
from datasets import load_dataset

class LLMModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipe = None

    def load_model(self, model_path=None):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        if model_path and os.path.exists(model_path):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, pad_token_id=0)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def query_model(self, input_text):
        try:
            torch.cuda.empty_cache()
            messages = [{"role": "user", "content": input_text}]
            
            outputs = self.pipe(
                messages,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            assistant_response = outputs[0]["generated_text"][-1]["content"]
            return assistant_response
            
        except torch.cuda.OutOfMemoryError:
            return "CUDA out of memory. Please try reducing max_new_tokens or batch size."
        except Exception as e:
            return f"An error occurred: {e}"

    def fine_tune(self, train_file, save_path="./fine_tuned_model"):
        dataset = load_dataset(
            'text',
            data_files=train_file,
            split='train'
        )

        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding="max_length",
                max_length=128  
            )

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        training_args = TrainingArguments(
            output_dir=save_path,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_datasets
        )

        trainer.train()

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

# model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
# llm_model = LLMModel(model_name)

# llm_model.load_model(model_path="./fine_tuned_model")

# Fine-tuning 
# llm_model.fine_tune("your_text_file.txt", save_path="./fine_tuned_model")

# response = llm_model.query_model("Explain the concept of neural networks.")
# print(response)
