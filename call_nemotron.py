import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from bitsandbytes import Int4Params

model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, quantize='int4', device_map='auto')

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,  
)

def query_model(input_text):
    try:
        torch.cuda.empty_cache()  
        messages = [{"role": "user", "content": input_text}]
        outputs = pipe(
            messages,
            max_new_tokens=128, 
            do_sample=False,
        )
        assistant_response = outputs[0]["generated_text"][-1]["content"]
        print(assistant_response)
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Please try reducing max_new_tokens or batch size.")
    except Exception as e:
        print(f"An error occurred: {e}")

while True:
    input_text = input("Enter your question (or 'exit' to stop): ")
    if input_text.lower() == 'exit':
        break
    response = query_model(input_text)
    print("Response:", response)
