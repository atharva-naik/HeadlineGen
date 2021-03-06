# -*- coding: utf-8 -*-
"""ProphetNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11uPpxZzoTofnHZsyDPaJr5dFKsgOmTD9

# Mount Drive
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

"""# Set up repo"""

cd drive/My Drive/Inter_IIT_Headline_Generation/HeadlineGen

#!mkdir models/

cd models/

#!mkdir ProphetNet/

cd ProphetNet

#!mkdir saved_models/

"""# Pull from repo"""

!git pull origin main

"""# Push to Repo"""

uname = "vmm221313"
!git config --global user.email '$uname@gmail.com'
!git config --global user.name '$uname'

from getpass import getpass
password = getpass('Password:')

!git status

!git add src/prepro/data_builder.py

!git commit -m 'Minor Bugfixes via Colab'  
!git push origin main

"""# Imports and Installs"""

!pip install datasets
!pip install transformers

!pip install git-python==1.0.3
!pip install sacrebleu==1.4.12
!pip install rouge_score

import datasets
import pandas as po
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from typing import Optional
from transformers import Trainer
from dataclasses import dataclass, field
from transformers import TrainingArguments
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup

"""# ProphetNet toy example"""

from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration

tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')

test_text = "An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, according to the state-run Brazilian news agency"

input_ids = tokenizer(test_text, return_tensors="pt").input_ids  # Batch size 1

#decoder_input_ids = tokenizer("An American woman", return_tensors="pt").input_ids  # Batch size 1

outputs = model.generate(input_ids=input_ids, num_beams=4, max_length=30, early_stopping=True)

print([tokenizer.decode(token, skip_special_tokens=True, clean_up_tokenization_spaces=False) for token in outputs])

"""# ProphetNet fine tuning"""

from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration

tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')

model.cuda()
'block print'

batch_size = 16  
encoder_max_length = 512
decoder_max_length = 128

pwd

RAW_DATA_PATH = "../../raw_data/Development Data/dev_data_article.xlsx"

df = po.read_excel(RAW_DATA_PATH)

df

df['Text'][:20]

df['Headline'][:20]

train_article_encodings = tokenizer(df['Text'][:10].tolist(), truncation=True, padding=True)
train_headline_encodings = tokenizer(df['Headline'][:10].tolist(), truncation=True, padding=True)

val_article_encodings = tokenizer(df['Text'][10:20].tolist(), truncation=True, padding=True)
val_headline_encodings = tokenizer(df['Headline'][10:20].tolist(), truncation=True, padding=True)

device='cuda'

class ArticleDataset(Dataset):
    def __init__(self, article_encodings, headline_encodings):
        self.article_encodings = article_encodings
        self.headline_encodings = headline_encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.article_encodings.items() if key!='token_type_ids'}
        for dec_key, dec_val in self.headline_encodings.items():
            if dec_key!='token_type_ids':
              item['decoder_'+dec_key] = torch.tensor(dec_val[idx]).to(device)
          
        item['labels'] = torch.tensor(self.headline_encodings['input_ids'][idx]).to(device) # decoder_input_ids

        return item

    def __len__(self):
        return len(self.article_encodings)

train_dataset = ArticleDataset(train_article_encodings, train_headline_encodings)
val_dataset = ArticleDataset(val_article_encodings, val_headline_encodings)

train_dataset[0].keys()

train_dataloader = DataLoader(train_dataset, batch_size=2)

BATCH_SIZE=1
gradient_accumulation_steps=1
weight_decay=0
epsilon=1e-8
learning_rate=2e-5
number_of_warmup_steps=0
num_train_epochs=1

t_total = len(train_dataloader) // gradient_accumulation_steps * 3

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=number_of_warmup_steps, num_training_steps=t_total)

model.zero_grad()
global_step = 0
for i in tqdm(range(num_train_epochs)):
  epoch_iterator = tqdm(train_dataloader, desc="Iteration")
  for step, inputs in enumerate(epoch_iterator):
      model.train()
      ouputs = model(**inputs)

      loss = ouputs[0]  
      loss.backward() 
      
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
      tr_loss += loss.item()
      
      if (step + 1) % gradient_accumulation_steps == 0:
          optimizer.step()
          scheduler.step()  # Update learning rate schedule
          model.zero_grad()
          global_step += 1
      
      # Save Model
      torch.save(model, 'saved_models/prophet_net_epoch{}.pt'.format(i))





