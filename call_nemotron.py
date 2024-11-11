import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def query_model(input_text):
    messages = [{"role": "user", "content": input_text}]
    tokenized_message = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    
    input_ids = tokenized_message['input_ids'].cuda()
    attention_mask = tokenized_message['attention_mask'].cuda()
    
    response_token_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    
    generated_tokens = response_token_ids[:, len(input_ids[0]):]
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return generated_text

while True:
    input_text = input("Enter your question (or 'exit' to stop): ")
    if input_text.lower() == 'exit':
        break
    response = query_model(input_text)
    print("Response:", response)
