from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B', dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
# prompt = '$1'
# prompt = 'What is the capital of France?'
# prompt = "How many r's are in the word 'strawberry'?"
## Give me a reasoning question
prompt = "If I have 3 apples and I give you 2, how many apples do I have left?"

out = model.generate(tokenizer(prompt, return_tensors='pt')['input_ids'], max_new_tokens=50, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
