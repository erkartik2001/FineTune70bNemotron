from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)



def query_model(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

while True:
    input_text = input("Enter your question (or 'exit' to stop): ")
    if input_text.lower() == 'exit':
        break
    response = query_model(input_text)
    print("Response:", response)


