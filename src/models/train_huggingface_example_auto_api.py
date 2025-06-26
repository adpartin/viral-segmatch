"""
Based on notebook from:
https://github.com/huggingface/transformers/tree/main/notebooks

Trains a binary classifier on protein sequences using huggingface ESM-2 and
the Trainer API.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from evaluate import load

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.torch_utils import determine_device


# Define compute_metrics function
metric = load('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


# ===================================
# Config
# ===================================
SEED = 42
TASK_NAME = 'huggingface_example'
CUDA_NAME = 'cuda:5'  # Specify GPU device
MODEL_CKPT = 'facebook/esm2_t6_8M_UR50D'  # Choose ESM-2 model
# MODEL_CKPT = 'facebook/esm2_t12_35M_UR50D'  # Choose ESM-2 model

NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

model_name = MODEL_CKPT.split('/')[-1]

# Set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Define paths
filepath = Path(__file__).resolve().parent
# data_dir = filepath / '../../data' / TASK_NAME
main_data_dir = filepath / '../../data'
datasets_dir = main_data_dir / 'datasets' / TASK_NAME
model_dir = filepath / '../../models'
output_dir = model_dir / (TASK_NAME + '_auto')

# Create output dir
os.makedirs(output_dir, exist_ok=True)

# Load data
train_seqs = np.load(datasets_dir / 'train_sequences.npy', allow_pickle=True).tolist()
test_seqs = np.load(datasets_dir / 'test_sequences.npy', allow_pickle=True).tolist()
train_labels = np.load(datasets_dir / 'train_labels.npy', allow_pickle=True)
test_labels = np.load(datasets_dir / 'test_labels.npy', allow_pickle=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_CKPT)

# Tokenized sequences is a dictionary with keys: input_ids, attention_mask
# input_ids is a list of token ids to be fed to the model
# train_tokenized = tokenizer(train_seqs) # original
# test_tokenized = tokenizer(test_seqs)   # original
train_tokenized = tokenizer(train_seqs, padding=True, truncation=True, return_tensors='pt')
test_tokenized = tokenizer(test_seqs, padding=True, truncation=True, return_tensors='pt')

# Convert tokenized sequences to Dataset
train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)

# Add labels to the dataset
train_dataset = train_dataset.add_column('labels', train_labels)
test_dataset = test_dataset.add_column('labels', test_labels)

print('Train_dataset: ', len(train_dataset))
print('Test_dataset:  ', len(test_dataset))

### Check tekonized sequences
print(f"train_dataset input_ids:      {train_dataset['input_ids'][0][:10]}")
print(f"train_dataset attention_mask: {train_dataset['attention_mask'][0][:10]}")
print(f"train_dataset label:          {train_dataset['labels'][0]}")

# ====================================
# Load ESM-2 Model
# ====================================
# Define training arguments
# https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/esm/modeling_esm.py#L1059
num_labels = max(train_labels.tolist() + test_labels.tolist()) + 1 # +1 since 0 can be a label
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CKPT, num_labels=num_labels)


# ====================================
# Train Model
# ====================================
# Training args
# github.com/huggingface/transformers/blob/v4.49.0/src/transformers/training_args.py#L224
args = TrainingArguments(
    output_dir = f'{model_name}-finetuned-localization',
    run_name = f'run-{model_name}-example',  # AP
    eval_strategy = 'epoch',
    save_strategy = 'epoch',
    learning_rate = LEARNING_RATE,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE,
    num_train_epochs = NUM_EPOCHS,
    weight_decay = WEIGHT_DECAY,
    load_best_model_at_end = True,
    metric_for_best_model = 'accuracy',
    push_to_hub = False,
)

# Create Trainer instance
trainer = Trainer(
    model = model,
    args = args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics,
)

# Train the model
start_time = time.time()
trainer.train()
print(f'\nRuntime {(time.time() - start_time)/60} minutes\n')

# Evaluate model and save results
# breakpoint()
test_results = trainer.evaluate()
pprint(f'Evaluation results: {test_results}')
with open(output_dir / 'test_results.json', 'w') as f:
    json.dump(test_results, f, indent=4)

# Compute predictions on test data and get the labels
test_preds_obj = trainer.predict(test_dataset)
test_preds = np.argmax(test_preds_obj.predictions, axis=1)

# Compute accuracy
accuracy = metric.compute(predictions=test_preds, references=test_labels)
print(f"Test Accuracy: {accuracy['accuracy']:.4f}")

# Print detailed classification report
print('\nClassification Report:')
print(classification_report(test_labels, test_preds,
                            target_names=['Cytosolic', 'Membrane'])
)

# Save raw predictions
test_res_df = pd.DataFrame({
    'seq': test_seqs,
    'label': test_labels,
    'pred': test_preds
})
test_res_df.to_csv(output_dir / 'test_predicted.csv', index=False)

# Save model
model_path = output_dir / 'best_model'
tokenizer_path = output_dir / 'tokenizer'
trainer.save_model(model_path) # this saves both model and tokenizer
tokenizer.save_pretrained(tokenizer_path)
print(f'Model saved: {model_path}')

print('Done.')
