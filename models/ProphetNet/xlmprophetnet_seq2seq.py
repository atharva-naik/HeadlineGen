import re
import nltk
import datasets
import numpy as np
import pandas as po
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import set_seed, TrainingArguments, Trainer, default_data_collator, AdamW, get_linear_schedule_with_warmup, Seq2SeqTrainer, Seq2SeqTrainingArguments

nltk.download('punkt')

set_seed(69)

metric = datasets.load_metric('rouge')

RAW_DATA_PATH = "../../raw_data/Development Data/dev_data_article.xlsx"

df = po.read_excel(RAW_DATA_PATH)
df

'. \n '.join(re.sub(r'[^a-zA-Z0-9\s\.]|(\n)', '', df['Text'].iloc[4].lower()).split('.'))

df['Text'] = df['Text'].apply(lambda x: '. \n '.join(re.sub(r'[^a-zA-Z0-9\s\.]|(\n)', '', x.lower()).split('.')))
df['Headline'] = df['Headline'].apply(lambda x: '. \n '.join(re.sub(r'[^a-zA-Z0-9\s\.]|(\n)', '', x.lower()).split('.')))

from transformers import XLMProphetNetConfig, XLMProphetNetTokenizer, XLMProphetNetForConditionalGeneration

config = XLMProphetNetConfig.from_pretrained('microsoft/xprophetnet-large-wiki100-cased')
tokenizer = XLMProphetNetTokenizer.from_pretrained('microsoft/xprophetnet-large-wiki100-cased')
model = XLMProphetNetForConditionalGeneration.from_pretrained('microsoft/xprophetnet-large-wiki100-cased')

#decoder_start_token?

# Get the column names for input/target.
dataset_columns = ('text', 'summary')
text_column = dataset_columns[0]
summary_column = dataset_columns[1]

# Temporarily set max_target_length for training.
max_source_length = 500
max_target_length = 64
padding = "max_length"

device='cuda'

df_formatted_train = df[['Text', 'Headline']][:3000].rename(columns={'Text': 'text', 'Headline': 'summary'})
df_formatted_val = df[['Text', 'Headline']][3000:].rename(columns={'Text': 'text', 'Headline': 'summary'})
df_formatted_train.to_csv('formatted_df_train.csv', index=False)
df_formatted_val.to_csv('formatted_df_val.csv', index=False)

train_dataset = datasets.load_dataset('csv', data_files='formatted_df_train.csv')['train']
val_dataset = datasets.load_dataset('csv', data_files='formatted_df_val.csv')['train']

train_dataset

val_dataset

data = {'train': train_dataset, 'validation': val_dataset}

column_names = data["train"].column_names
column_names

prefix=""

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[summary_column]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
            load_from_cache_file=False,
        )

eval_dataset = val_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
            load_from_cache_file=False,
        )

label_pad_token_id = -100 
data_collator = default_data_collator

metric_name = "rouge"
metric = datasets.load_metric(metric_name)

df_formatted_train.iloc[4]['text']

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    if metric_name == "rouge":
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    else:
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

"""## Train"""

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=1,   # batch size per device during training
    per_device_eval_batch_size=1,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

model.cuda()
'block print'

checkpoint = None
train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model()  # Saves the tokenizer too for easy upload

train_result

metrics = train_result.metrics
max_train_samples = len(train_dataset)

metrics["train_samples"] = min(max_train_samples, len(train_dataset))

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

"""## Evaluate"""
"""
checkpoint=None
metrics = trainer.evaluate(max_length=max_target_length, num_beams=4, metric_key_prefix="eval", resume_from_checkpoint=checkpoint)
metrics["eval_samples"] = len(eval_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
"""


