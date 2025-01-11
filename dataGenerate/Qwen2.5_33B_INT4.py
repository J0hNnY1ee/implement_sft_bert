'''
Author: J0hNnY1ee j0h1eenny@gmail.com
Date: 2025-01-08 22:37:38
LastEditors: J0hNnY1ee j0h1eenny@gmail.com
LastEditTime: 2025-01-08 23:56:41
FilePath: /implement_sft_bert/dataGenerate/Qwen2.5_33B_INT4.py
Description: 

Copyright (c) 2025 by J0hNnY1ee j0h1eenny@gmail.com, All Rights Reserved. 
'''
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "model/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
