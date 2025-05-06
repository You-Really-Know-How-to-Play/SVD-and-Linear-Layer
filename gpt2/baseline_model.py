# define an untrained baseline gpt2 model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn as nn
from transformers import GPT2Config

class GPT2BaselineModel(nn.Module):
    # init with untrained GPT2 model
    def __init__(self, config, tokenizer):
        super(GPT2BaselineModel, self).__init__()
        self.config = config
        self.model = GPT2LMHeadModel(config)
        self.tokenizer = tokenizer
        self.model.resize_token_embeddings(len(tokenizer))
    
    def forward(self, input_ids, attention_mask=None, labels=None, past_key_values=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels, past_key_values=past_key_values)
        return outputs
    

# Test the model

if __name__ == "__main__":
    cache_path = "D:/huggingface/cache/"
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2', cache_dir=cache_path)
    config = GPT2Config()
    model = GPT2BaselineModel(config, tokenizer)

    # Test the model with a sample input
    input_text = "Once upon a time in a land far away"
    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    outputs = model(input_ids, attention_mask=attention_mask)
    # print the keys of the output
    print("Output keys:", outputs.keys())

    output_text = tokenizer.decode(outputs.logits.argmax(dim=-1).squeeze().tolist())
    print("Output text:", output_text)
