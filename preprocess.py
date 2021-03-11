import os 
import re 
import copy
import json
import tqdm
import emoji
import sklearn 
import pandas as pd
from string import ascii_letters, digits

def lower(text):
    return text.lower()

def strip(text):
    return text.strip()

def filter_charset(text):
    # hindi+englishset+.+digits
    return text

def remove_newline(text):
    return text.replace("\n", " ")

def insert_newline(text):
    return "\n".join([i.strip() for i in text.split(".")])

def remove_urls(text):
    return re.sub(r'http\S+', '', text)

def remove_punctuation(text):
    for punct in '''"',:`;%@!|\&()#$+-_?<>*=~{}[]''':
        text = text.replace(punct, " ")

    return text

def identity(text):
    return text

VALID_FILTERS = {"lower":lower,  
                 "strip":strip,
                 "remove_urls":remove_urls,
                 "remove_punctuation":remove_punctuation,
                 "remove_newline":remove_newline,
                 "insert_newline":insert_newline}


class ProcessText:
    @classmethod
    def Sequential(cls, filts):
        obj = cls()
        obj.filts = []

        # c.valid_filts = VALID_FILTERS
        for filt in filts:
            obj.filts.append(obj.valid_filts.get(filt, identity))

        return obj

    def __init__(self):
        self.filts = [] 
        self.valid_filts = VALID_FILTERS
        
    def add(self, filt):
        self.filts.append(filt)

    def run(self, text):
        for filt in self.filts:
            text = filt(text)

        return text 


def process_dataset(path, pipeline, seed, valid_size=1/8, test_size=1/8, target=['Text', 'Headline'], keep=['Text', 'Headline'], rename=['text', 'summary']):
    if path.endswith(".csv"):
        dataset = pd.read_csv(path)
    elif path.endswith(".xlsx"):
        dataset = pd.read_excel(path)
    else:   
        raise(TypeError("\x1b[31mFile format not valid\x1b[0m"))

    dataset = dataset[keep]
    map_dict = {k:v for k,v in zip(keep, rename)}
    
    text_processor = ProcessText.Sequential(pipeline)
    proc_datatset = []
    
    for line in tqdm.tqdm(dataset.to_dict("records")):
        for key in target:
            line[key] = text_processor.run(line[key])
        proc_datatset.append(line)
    total = len(dataset)

    proc_datatset = pd.DataFrame(proc_datatset)
    proc_datatset = proc_datatset.sample(frac=1, random_state=seed).reset_index(drop=True)
    proc_datatset.rename(columns=map_dict, inplace=True, errors='raise')
    test_size = int(total*test_size)
    valid_size = int(total*valid_size)
    train_size = total - test_size - valid_size
    
    proc_datatset = proc_datatset.to_dict('records')
    train = proc_datatset[ : train_size]
    val = proc_datatset[train_size : train_size + valid_size]
    test = proc_datatset[train_size + valid_size : ]

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    val = pd.DataFrame(val)

    return train, val, test

if __name__ == "__main__":
    df = pd.read_excel("/home/atharva/interiit/HeadlineGen/raw_data/Development Data/dev_data_article.xlsx")
    print(df.head())
    pipeline = ["strip", "remove_newline", "remove_url", "remove_punctuation", "insert_newline", "lower"]
    # pipeline = ["identity"]
    train, val, test = process_dataset("/home/atharva/interiit/HeadlineGen/raw_data/Development Data/dev_data_article.xlsx", pipeline, 69)
    print(train.head())
    print(val.head())
    print(test.head())
# if TRANSLATE:
#     index = 0
#     def translate(text):
#         return text

#     for headline, article in zip(headlines, articles):
#         trans_dataset[index]['en_Headline'] = translate(headline)
#         trans_dataset[index]['en_Text'] = translate(article)
#         index += 1