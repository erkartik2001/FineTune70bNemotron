from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from transformers import BitsAndBytesConfig
import os
import torch

class LlamaModel:
    def __init__(self, model_name, token):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.model_name = model_name
        self.token = token
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.token)

        if self.tokenizer.pad_token is None:
            print("Adding a padding token to the tokenizer...")
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=self.bnb_config,
            token=self.token
        )

        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def process_text(self, large_text, chunk_size=512):
        """
        Splits a large text string into smaller chunks suitable for fine-tuning.

        Args:
            large_text (str): The large input text to process.
            chunk_size (int): Maximum token length for each chunk.

        Returns:
            list: A list of dictionaries containing text chunks.
        """
        print("Processing large text into fine-tuning chunks...")
        tokens = self.tokenizer(large_text, return_tensors="pt", truncation=False).input_ids[0]
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        processed_data = [
            {"text": self.tokenizer.decode(chunk, skip_special_tokens=True)}
            for chunk in chunks
        ]
        print(f"Processed {len(processed_data)} chunks.")
        return processed_data

    def fine_tune(self, training_data, output_dir="./fine_tuned_model", epochs=3, batch_size=4, learning_rate=5e-5):
        """
        Fine-tunes the model on the provided training data.

        Args:
            training_data (list): List of dictionaries with "text" as the key for fine-tuning.
            output_dir (str): Directory to save the fine-tuned model.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
        """
        if self.tokenizer is None or self.model is None:
            raise ValueError("Model and tokenizer must be loaded before fine-tuning. Call load_model() first.")

        print("Preparing dataset for fine-tuning...")
        dataset = Dataset.from_list(training_data)

        def tokenize_function(example):
            return self.tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        print("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=10,
            fp16=torch.cuda.is_available(),
        )

        print("Starting fine-tuning...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        trainer.train()

        print("Saving fine-tuned model...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Fine-tuned model saved at {output_dir}")

    def generate_text(self, prompt, max_length=100, num_return_sequences=1):
        """
        Generates text based on the provided prompt.

        Args:
            prompt (str): Input prompt to generate text.
            max_length (int): Maximum length of the generated text.
            num_return_sequences (int): Number of sequences to generate.

        Returns:
            list: List of generated text sequences.
        """
        if self.tokenizer is None or self.model is None:
            raise ValueError("Model and tokenizer must be loaded before generating text. Call load_model() first.")

        print("Generating text...")
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

if __name__ == "__main__":
    model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
    token = "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"

    llama = LlamaModel(model_name, token)
    llama.load_model()

    large_text = """
    This is an example of a very large text. It spans multiple sentences and paragraphs. 
    The purpose is to simulate real-world text data for fine-tuning a language model.
    Here is some additional text to make it even longer.
    """

#     # Process the large text into chunks
#     training_data = llama.process_text(large_text, chunk_size=512)

#     # Fine-tune the model
#     llama.fine_tune(training_data, output_dir="./fine_tuned_llama")

    generated_text = llama.generate_text("The fine-tuned model excels at", max_length=50)
    print("\nGenerated Text:")
    print(generated_text)