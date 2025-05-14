# Test the baseline model
import random
import numpy as np
import torch
from transformers import GPT2Tokenizer
from baseline_model import GPT2BaselineModel
from transformers import GPT2Config, GPT2LMHeadModel


cache_path = "D:/huggingface/cache/"
tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2', cache_dir=cache_path)

# load the model weights
model_path = "gpt2/baseline_gpt2_model/best_model.pt"
#config = GPT2Config(n_embd=384, n_layer=3, n_head=6, n_positions=256, attn_pdrop=0.2, embd_pdrop=0.2)
# Load the model
model = torch.load(model_path, weights_only=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Test the model with a sample input
input_text = "To be or not to be,"

inputs = tokenizer(input_text, return_tensors='pt').to(device)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# generate output
model.eval()
with torch.no_grad():
    output_text = model.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50, temperature=0.7, do_sample=True, top_k = 50, top_p=0.95, num_return_sequences=5)
    #output_text = tokenizer.decode(output_text[0], skip_special_tokens=True)
    output_text = tokenizer.batch_decode(output_text, skip_special_tokens=True)
print("Input text:", input_text)
for i, text in enumerate(output_text):
    print(f"Output text {i+1}:", text)

