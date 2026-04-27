# FineTune70bNemotron

A Python-based toolkit for fine-tuning the NVIDIA Llama 3.1 Nemotron 70B language model with 4-bit quantization and efficient training support.

## Features

- **4-bit Quantization**: Uses BitsAndBytes configuration for memory-efficient model loading
- **Text Processing**: Intelligently chunks large texts into token-based segments for fine-tuning
- **Fine-tuning**: Full training pipeline with configurable epochs, batch size, and learning rate
- **Text Generation**: Generate text from fine-tuned models with customizable parameters
- **GPU Support**: Automatic device detection and CUDA optimization

## Installation

```bash
pip install transformers datasets torch bitsandbytes
```

## Quick Start

```python
from llm_model import LlamaModel

# Initialize model
llama = LlamaModel("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", token="your_hf_token")

# Load model and tokenizer
llama.load_model()

# Process your training text
training_data = llama.process_text(your_large_text, chunk_size=512)

# Fine-tune the model
llama.fine_tune(training_data, output_dir="./fine_tuned_llama", epochs=3)

# Generate text
outputs = llama.generate_text("Your prompt here", max_length=100)
print(outputs)
```

## Key Methods

- **`load_model()`**: Loads the tokenizer and model with 4-bit quantization
- **`process_text(text, chunk_size)`**: Splits large text into fine-tuning chunks
- **`fine_tune(training_data, output_dir, epochs, batch_size, learning_rate)`**: Fine-tunes the model
- **`generate_text(prompt, max_length, num_return_sequences)`**: Generates text completions

## Requirements

- PyTorch with CUDA support
- Hugging Face `transformers` and `datasets` libraries
- BitsAndBytes for quantization
- GPU with sufficient memory (recommended: 24GB+)
