from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# prompt = "What is the capital of France?"
# prompt = "How many r's are in the word 'strawberry'?"
prompt = "If I have 3 apples and I give you 2, how many apples do I have left?"

# Format as chat (instruct model expects this)
messages = [{"role": "user", "content": prompt}]
chat_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(chat_text, return_tensors="pt")
prompt_len = inputs["input_ids"].shape[1]
out = model.generate(
    **inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id
)
# Only print the generated response (skip the prompt template)
response = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
print(f"Q: {prompt}")
print(f"A: {response}")
