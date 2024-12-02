from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
save_directory = "./llama-70B"
access_token = "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"

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_directory,token=access_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_directory,token=access_tokenv)

print(f"Model and tokenizer saved to {save_directory}")
