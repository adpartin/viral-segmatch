"""
Got this notebook from:
https://github.com/huggingface/transformers/tree/main/notebooks

Train a binary classifier for protein sequences using huggingface ESM-2
model using explicit training loop.


You are an experienced software developer with strong knowledge of deep learning. You're very detail-oriented and critical thinker, striving for well-structured code. I need your with the following:

I have two implementations:
1. the auto approach (AutoModelForSequenceClassification) --> accuracy 0.94
2. the exp (explicit) approach (EsmModel and a EsmClassificationHead on top) --> accuracy 0.85

Both models run on the same machine (remote GPU cluster). I expect to see the same performance.

I started with the auto approach from huggingface examples repo. The reason I explore the explicit approach is that I want more control on the entire pipeline, so that later, I want to extract an embedding layer from the ESM model and use it for some other downstream task. Thus, I do this comparison to make sure I properly use the model.

Can I provide you both approaches?
"""
import os
import random
import sys
import time
from pathlib import Path
from pprint import pprint
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    EsmConfig,
    EsmModel,
)
from evaluate import load

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.torch_utils import determine_device


class ProteinDataset(Dataset):
    """Dataset for binary protein classification."""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class TokenizedProteinDataset(Dataset):
    """Dataset for protein classification with tokenized sequences."""
    def __init__(self, tokenizer, sequences, labels=None):
        self.tokenizer = tokenizer
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

        # Tokenize sequences once during dataset creation (avoids repeated
        # tokenization)
        self.encodings = self.tokenizer(
            sequences, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        # Each sample returns input_ids, attention_mask, and labels
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

class EsmClassificationHead(nn.Module):
    """Head for sentence-level classification tasks.
    This is how the the classification head is implemented in Huggingface
    EsmForSequenceClassification. See here:
    - github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/esm/modeling_esm.py#L1073
    - github.com/huggingface/transformers/blob/main/src/transformers/models/esm/modeling_esm.py#L1230
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ProteinClassifier(nn.Module):
    """ Implementation of EsmForSequenceClassification in Huggingface.
    github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/esm/modeling_esm.py#L1066
    """
    # def __init__(self, config):
    def __init__(self, model_ckpt, num_labels=2):
        super().__init__()
        # github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/esm/modeling_esm.py#L1072C1-L1073C56
        # EsmModel by default is add_pooling_layer=True:
        #    github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/esm/modeling_esm.py#L777
        # EsmForSequenceClassification by default is add_pooling_layer=False:
        #    github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/esm/modeling_esm.py#L1072
        self.esm = EsmModel.from_pretrained(model_ckpt, add_pooling_layer=False)
        self.config = self.esm.config
        self.config.num_labels = num_labels
        self.classifier = EsmClassificationHead(self.config)

    def forward(self, input_ids, attention_mask=None):
        # github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/esm/modeling_esm.py#L1103
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        # sequence_output = outputs[0] # what about this line??
        logits = self.classifier(outputs.last_hidden_state)
        return logits

# Define compute_metrics function
metric = load('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


# ===================================
# Config
# ===================================
TASK_NAME = 'cyto_vs_memb'
CUDA_NAME = 'cuda:6'  # Specify GPU device
MODEL_CKPT = 'facebook/esm2_t6_8M_UR50D'  # Choose ESM-2 model
# MODEL_CKPT = 'facebook/esm2_t12_35M_UR50D'  # Choose ESM-2 model
SEED = 42  # For reproducibility

NUM_EPOCHS = 5  # Number of training epochs
BATCH_SIZE = 8  # Increase batch size for efficiency
LEARNING_RATE = 2e-5  # Optimizer learning rate
WEIGHT_DECAY = 0.01  # Weight decay

model_name = MODEL_CKPT.split('/')[-1]

# Set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Define paths
filepath = Path(__file__).resolve().parent
data_dir = filepath / '../../data' / TASK_NAME
model_dir = filepath / '../../models'
output_dir = model_dir / TASK_NAME

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load data
train_sequences = np.load(data_dir / 'train_sequences.npy', allow_pickle=True).tolist()
test_sequences = np.load(data_dir / 'test_sequences.npy', allow_pickle=True).tolist()
train_labels = np.load(data_dir / 'train_labels.npy', allow_pickle=True)
test_labels = np.load(data_dir / 'test_labels.npy', allow_pickle=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)

# Create dataset
# train_dataset = ProteinDataset(train_sequences, train_labels)
# test_dataset = ProteinDataset(test_sequences, test_labels)
train_dataset = TokenizedProteinDataset(tokenizer, train_sequences, train_labels)
test_dataset  = TokenizedProteinDataset(tokenizer, test_sequences,  test_labels)

# Create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # TODO: shuffle=False
test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print('Train_dataset: ', len(train_dataset))
print('Test_dataset:  ', len(test_dataset))

### Check dataloader
# batch = next(iter(train_dataloader))
# print('dataloader batch', batch.keys())
# print(batch['input_ids'].shape)
# print(batch['attention_mask'].shape)
# print(batch['labels'].shape)
# print(batch['input_ids'][0])
# print(batch['attention_mask'][0])
# # print(batch['input_ids'][1])
# # print(batch['attention_mask'][1])

### Manually tokenize a sequence and compare with dataset
# manual_token = tokenizer(train_sequences[0], return_tensors='pt', padding=True, truncation=True)
# data_sample = train_dataset[0]
# assert torch.equal(
#     manual_token['input_ids'].squeeze(0),
#     data_sample['input_ids'][:manual_token['input_ids'].size(1)]
# ), "Tokenization mismatch!"
# assert torch.equal(
#     manual_token['attention_mask'].squeeze(0),
#     data_sample['attention_mask'][:manual_token['attention_mask'].size(1)]
# ), "Tokenization mismatch!"
# print("Tokenization is consistent!")


# ====================================
# Load ESM-2 Model
# ====================================
# Use EsmClassificationHead to follow the same structure as
# EsmForSequenceClassification
# Load model config
"""
Note! You can print the returned object and observe the config.
When you load a pre-trained model like facebook/esm2_t12_35M_UR50D using
EsmConfig.from_pretrained(MODEL_CKPT), you're not getting the default configs
Instead, you're getting the specific configuration that was used when that
particular model was trained.
You'll also see the param "architectures": ["EsmForMaskedLM"]. This indicates
the primary task the pre-trained model was trained for, which in this case of
esm2_t12_35M_UR50D is Masked Language Modeling (MLM).
"""
# Initialize classifier and move it to device
device = determine_device(CUDA_NAME)
classifier = ProteinClassifier(MODEL_CKPT, num_labels=2).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(classifier.parameters(), lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY)


# ====================================
# Train Model
# ====================================
# breakpoint()
print('\nStarting training ...')

best_loss = float('inf')
best_model_state = None

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    classifier.train()
    total_loss = 0.0

    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}',
        dynamic_ncols=True, leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = classifier(**inputs)

        # Backward pass
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar with loss value
        progress_bar.set_postfix(loss=f'{loss.item():.5f}')

    avg_train_loss = total_loss / len(train_dataloader)

    # # Validation step
    # classifier.eval()
    # val_loss = 0
    # with torch.no_grad():
    #     for batch in val_dataloader:
    #         inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
    #         labels = batch['labels'].to(device)
    #         outputs = classifier(**batch)
    #         loss = outputs.loss
    #         val_loss += loss.item()

    # avg_val_loss = val_loss / len(val_dataloader)

    # # Save best model based on validation loss
    # if avg_val_loss < best_loss:
    #     best_loss = avg_val_loss
    #     best_model_state = model.state_dict()  # Save model parameters

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader):.4f}')
print(f'runtime {(time.time() - start_time)/60} minutes')
del batch, inputs, labels, outputs, loss

# Restore best model state at the end
if best_model_state:
    classifier.load_state_dict(best_model_state)
    print('Best model restored based on validation loss.')


# ===================================
# Evaluate model on test set
# ===================================
# breakpoint()
test_preds, all_test_labels = [], []

classifier.eval()
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc='Evaluating'):
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)

        # Forward pass (with esm_model and classifier)
        logits = classifier(**inputs)  # under torch.no_grad()
        preds = torch.argmax(logits, dim=1)

        # Store preds and labels
        test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

# Compute accuracy
accuracy = metric.compute(predictions=test_preds, references=all_test_labels)
print(f"Test Accuracy: {accuracy['accuracy']:.4f}")

# Print detailed classification report
print('\nClassification Report:')
print(classification_report(all_test_labels, test_preds, 
                            target_names=['Cytosolic', 'Membrane'])
)

# Save raw predictions
test_res_df = pd.DataFrame({
    'seqs': test_sequences,
    'labels': all_test_labels,
    'preds': test_preds
})
test_res_df.to_csv(output_dir / 'test_predicted.csv', index=False)

# Save model
model_path = output_dir / 'model.pt'
tokenizer_path = output_dir / 'tokenizer'
torch.save(classifier, model_path)
torch.save(classifier.state_dict(), output_dir / 'model_state_dict.pth')
tokenizer.save_pretrained(tokenizer_path)
print(f'Model saved to {model_path}')
print('Done!')
