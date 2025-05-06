# prepare the dataset for mini-GPT model to generate text in the style of William Shakespeare.
"""
Dataset Preparation: Use the dataset ”data/tinyshakespeare.txt”
from the codebase as the source dataset. Split the dataset into training,
validation, and testing sets with a ratio of 80%, 10%, and 10%, respectively.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm
import random

def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())

class ShakespeareDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, all_data):
        # Tokenize the data
        tokenized_data = self.tokenizer(all_data, return_tensors='pt', padding=True, truncation=True)
        
        # Extract input_ids and attention masks
        input_ids = torch.LongTensor(tokenized_data['input_ids'])
        attention_mask = torch.LongTensor(tokenized_data['attention_mask'])
        
        # Create a dictionary to hold the batched data
        batched_data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        return batched_data
    
"""
The Shakespeare dataset is a text file with the structrue:
for paragraph in corpus:
    "Name of the character":
    "Some conversation text"
    empty line
We only use the conversation text for training.
"""

def load_dataset(file_path):
    paraphrase_data = []
    current_paragraph = ""
    last_line_is_empty = False
    first_line = True
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            # Ignore the first line
            if first_line:
                first_line = False
                continue
            line = line.strip()
            if line == "": # append the recorded paragraph to the dataset
                if current_paragraph:
                    paraphrase_data.append(preprocess_string(current_paragraph))
                    current_paragraph = ""
                last_line_is_empty = True
                continue
            # ignore the character names
            if last_line_is_empty:
                last_line_is_empty = False
                continue
            # if normal line, append to the current paragraph
            current_paragraph += line + " "
        # Append the last paragraph if it exists
        if current_paragraph:
            paraphrase_data.append(preprocess_string(current_paragraph))
    
    return paraphrase_data

# test 
if __name__ == "__main__":
    # download the gpt2 model
    # The variable cache_path is used to set the cache directory, please modify it according to your own environment.
    cache_path = "D:/huggingface/cache/"
    model = GPT2Model.from_pretrained('openai-community/gpt2', cache_dir=cache_path)
    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2', cache_dir=cache_path)
    # Load the dataset
    file_path = "gpt2/data/tinyshakespeare.txt"
    dataset = load_dataset(file_path)
    print(f"Loaded {len(dataset)} examples from {file_path}")
    # Split the dataset into train, validation, and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    # Suffle the dataset
    random.seed(5)
    random.shuffle(dataset)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    # Create the dataset objects
    train_dataset = ShakespeareDataset(train_dataset, tokenizer)
    val_dataset = ShakespeareDataset(val_dataset, tokenizer)
    test_dataset = ShakespeareDataset(test_dataset, tokenizer)
    # Print the first example
    print("Train dataset examples:")
    for i in range(1):
        print(train_dataset[i])
    print("Validation dataset examples:")
    for i in range(1):
        print(val_dataset[i])
    print("Test dataset examples:")
    for i in range(1):
        print(test_dataset[i])
    # test the collate function
    print("Train dataset collate function:")
    train_dataset_collate = train_dataset.collate_fn(train_dataset[:2])
    print(train_dataset_collate)
    # find the max length of the dataset
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=train_dataset.collate_fn)
    max_length = 0
    # use tqdm
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids']
        max_length = max(max_length, input_ids.shape[1])
    print(f"Max length of the dataset: {max_length}") # 793
    # print padding token id and vocab size
    print(f"Padding token id: {tokenizer.pad_token_id}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    

   
