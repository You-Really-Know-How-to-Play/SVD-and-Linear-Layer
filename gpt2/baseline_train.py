# Define the train process of the baseline GPT-2 model
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from data_preparation import ShakespeareDataset, load_dataset
from baseline_model import GPT2BaselineModel
from transformers import GPT2Config, get_linear_schedule_with_warmup

def seed_everything(seed=5):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

# train the model on next-token prediction

def train_baseline_gpt2(model, train_dataloader, dev_dataloader, optimizer, scheduler, args):
    model.train()
    train_loss_history = []
    dev_loss_history = []
    best_dev_loss = float('inf')
    best_dev_epoch = 0
    current_patience = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        epoch_train_loss = 0.0
        # training
        model.train()
        with tqdm(train_dataloader, desc="Training", unit="batch") as train_dataloader:
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)

                # shift the input ids to the right
                labels = input_ids[:, 1:].clone()
                input_ids = input_ids[:, :-1]
                attention_mask = attention_mask[:, :-1]

                # forward pass
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_train_loss += loss.item()

                # update the progress bar
                train_dataloader.set_postfix(loss=loss.item())
                train_dataloader.update(1)

        epoch_train_loss /= args.train_size
        train_loss_history.append(epoch_train_loss)
        # validation
        model.eval()
        print("Validation...")
        epoch_dev_loss = 0.0
        with torch.no_grad():
            with tqdm(dev_dataloader, desc="Validation", unit="batch") as dev_dataloader:
                for batch in dev_dataloader:
                    input_ids = batch['input_ids'].to(args.device)
                    attention_mask = batch['attention_mask'].to(args.device)
                    # shift the input ids to the right
                    labels = input_ids[:, 1:].clone()
                    input_ids = input_ids[:, :-1]
                    attention_mask = attention_mask[:, :-1]
                    # forward pass
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    epoch_dev_loss += loss.item()
                    # update the progress bar
                    dev_dataloader.set_postfix(loss=loss.item())
                    dev_dataloader.update(1)
        epoch_dev_loss /= args.dev_size
        dev_loss_history.append(epoch_dev_loss)
        # early stopping
        if epoch_dev_loss < best_dev_loss:
            best_dev_loss = epoch_dev_loss
            best_dev_epoch = epoch
            current_patience = 0
            # save the model
            model.model.save_pretrained(args.output_dir)
        else:
            current_patience += 1
            if current_patience >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        print(f"Epoch {epoch + 1}/{args.epochs} - Train loss: {epoch_train_loss:.4f} - Dev loss: {epoch_dev_loss:.4f} - Best dev loss: {best_dev_loss:.4f} - Best dev epoch: {best_dev_epoch + 1} - Current patience: {current_patience}/{args.patience}")

    return train_loss_history, dev_loss_history

def compute_loss(outputs_logits, correct_labels, loss_fct):
    # Compute the loss
    loss = loss_fct(outputs_logits.view(-1, outputs_logits.size(-1)), correct_labels.view(-1))
    return loss

class train_args:
    def __init__(self, train_size, dev_size, test_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 4
        self.batch_size = 2
        self.learning_rate = 1e-5
        self.patience = 3
        self.output_dir = "gpt2/baseline_gpt2_model/"
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.train_size = train_size
        self.dev_size = dev_size
        self.test_size = test_size

if __name__ == "__main__":
    # Set the seed for reproducibility
    seed_everything(5)

    # Load the dataset
    file_path = "gpt2/data/tinyshakespeare.txt"
    dataset = load_dataset(file_path)
    print(f"Loaded {len(dataset)} examples from {file_path}")

    # Split the dataset into train, validation, and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Load the training arguments
    args = train_args(train_size, val_size, test_size)
    
    print(f"Using device: {args.device}")
    # Load the tokenizer and model
    cache_path = "D:/huggingface/cache/"
    tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2', cache_dir=cache_path)
    config = GPT2Config(n_layer=4, n_head=4, n_embd=384)
    model = GPT2BaselineModel(config, tokenizer)
    model.to(args.device)



    # Suffle the dataset
    random.shuffle(dataset)
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create dataloaders
    train_dataset = ShakespeareDataset(train_dataset, tokenizer)
    val_dataset = ShakespeareDataset(val_dataset, tokenizer)
    test_dataset = ShakespeareDataset(test_dataset, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    # Create the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = (len(train_dataloader) // args.batch_size + 1) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Start training
    print("Starting training...")
    train_loss_history, dev_loss_history = train_baseline_gpt2(model, train_dataloader, val_dataloader, optimizer, scheduler, args)
    print("Training finished.")

    # Print the training and validation loss history
    print("Training loss history:", train_loss_history)
    print("Validation loss history:", dev_loss_history)


  