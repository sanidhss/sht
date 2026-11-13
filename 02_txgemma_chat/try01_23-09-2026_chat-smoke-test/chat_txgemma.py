import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "google/txgemma-27b-chat"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", dtype=torch.bfloat16)

messages = [
    {"role": "user", "content": "What are the key ADMET liabilities to consider for an orally dosed small molecule kinase inhibitor?"},
]

inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("--- reply ---")
print(reply)
print("--- GPU memory ---")
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} {torch.cuda.memory_allocated(i) / 1e9:.1f} GB allocated")
