# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# prompt = "How many r in strawberry?"
# messages = [{"role": "user", "content": prompt}]

# tokenized_message = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
# response_token_ids = model.generate(tokenized_message['input_ids'].cuda(),attention_mask=tokenized_message['attention_mask'].cuda(),  max_new_tokens=4096, pad_token_id = tokenizer.eos_token_id)
# generated_tokens =response_token_ids[:, len(tokenized_message['input_ids'][0]):]
# generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
# print(generated_text)

from transformers import pipeline
import torch

model_id = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
    do_sample=False,
)
assistant_response = outputs[0]["generated_text"][-1]["content"]
print(assistant_response)
