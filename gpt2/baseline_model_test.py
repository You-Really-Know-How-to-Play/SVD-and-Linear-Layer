# Test the baseline model
import random
import numpy as np
import torch
from transformers import GPT2Tokenizer
from baseline_model import GPT2BaselineModel
from transformers import GPT2Config

config = GPT2Config(n_layer=4, n_head=4, n_embd=384)
cache_path = "D:/huggingface/cache/"
tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2', cache_dir=cache_path)

model = GPT2BaselineModel(config, tokenizer)
# load the model weights
model_path = "gpt2/baseline_gpt2_model/"
model.model.from_pretrained(model_path, config=config)

# Test the model with a sample input
input_text = "To be or not to be,"
print("Input text:", input_text)
inputs = tokenizer(input_text, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
outputs = model(input_ids, attention_mask=attention_mask)
# decode the output
output_text = tokenizer.decode(outputs.logits.argmax(dim=-1).squeeze().tolist())
print("Output text:", output_text)
