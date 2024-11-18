import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

class LLMModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipe = None

    def load_model(self):
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

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
                max_new_tokens=128,  
                do_sample=False,    
            )
            
            assistant_response = outputs[0]["generated_text"][-1]["content"]
            return assistant_response
            
        except torch.cuda.OutOfMemoryError:
            return "CUDA out of memory. Please try reducing max_new_tokens or batch size."
        except Exception as e:
            return f"An error occurred: {e}"

# model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
# llm_model = LLMModel(model_name)
# llm_model.load_model()  
