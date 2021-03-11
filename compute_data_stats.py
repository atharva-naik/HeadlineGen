import os 
import re 
import copy
import json
import tqdm
import pprint
import sklearn 
import pandas as pd
from sklearn.model_selection import train_test_split


for lib in ["emoji", "fasttext", "google_trans_new"]:
    try: 
        exec(f"import {lib}")
    except ImportError:
        os.system(f"pip install {lib}")
from google_trans_new import google_translator

DATASET_PATH = "raw_data/Development Data/dev_data_article.xlsx"
TRANSLATED_PATH = "/raw_data/Translated Data/"
PROCESSED_PATH = "raw_data/Processed Data"
TRANSLATE = False
STATS_PATH = "stats.json"
TRANSLATOR = google_translator()

dataset = pd.read_excel(DATASET_PATH)
trans_dataset = copy.deepcopy(dataset).to_dict("records")
articles = list(dataset['Text'])
headlines = list(dataset['Headline'])

def lower(text):
    return text.lower()

def remove_urls(text):
    return text

def remove_punctuation(text):
    result = ""
    for letter in text:
        if letter not in '''".',:`;''':
            result += letter
    
    return result

def identity(text):
    return text

VALID_FILTERS = {"lower":lower, "remove_punctuation":remove_punctuation}


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

text_processor = ProcessText.Sequential(["strip",
                                     "remove_urls", 
                                     "remove_punctuation"])
stats = {"total_lines":0,
         "total_headlines":0,
         "max_lines":0,
         "min_lines":100000,
         "avg_lines":0, 
         "max_line_len":0, 
         "min_line_len":100000,
         "avg_line_len":0, 
         "en_lines":0, 
         "hi_lines":0, 
         "eh_lines":0,
         "max_headline_len":0,
         "min_headline_len":100000,
         "avg_headline_len":0,
         "en_headlines":0,
         "hi_headlines":0,
         "eh_headlines":0}


try:
    open('/home/atharva/lid.176.bin', "rb")
except FileNotFoundError:
    os.system("wget -O /home/atharva/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
PRETRAINED_MODEL_PATH = '/home/atharva/lid.176.bin'
model = fasttext.load_model(PRETRAINED_MODEL_PATH)

def translate(text):
    return TRANSLATOR.translate(text, lang_tgt='en').strip()

if True:
    total_lines = 0
    total_words = 0

    for j,article in enumerate(tqdm.tqdm(articles)):
        article_lines = 0
        proc_article = ''
        for i,para in enumerate(article.split("\n")):
            for line in para.split("."):
                line = line.strip().strip("\n")
                if line != "":
                    article_lines += 1
                    proc_article += text_processor.run(line) + ' . '
                    num_words = len(line.split())

                    lang = model.predict([line])[0][0][0].split("__")[-1]
                    if lang == 'hi':
                        stats["hi_lines"] += 1
                        if TRANSLATE:
                            line = translate(line)
                    elif lang == 'en':
                        stats["en_lines"] += 1
                    else:
                        stats["eh_lines"] += 1
                        if TRANSLATE:
                            line = translate(line)
                    
                    total_words += num_words
                    stats["max_line_len"] = max(num_words, stats["max_line_len"])
                    stats["min_line_len"] = min(num_words, stats["min_line_len"])

        total_lines += article_lines
        stats["max_lines"] = max(article_lines, stats["max_lines"])
        stats["min_lines"] = min(article_lines, stats["min_lines"])
        proc_article = proc_article.strip().strip("\n")
        articles[j] = proc_article.replace("\n", "")

    stats["total_lines"] = total_lines
    stats["avg_line_len"] = total_words/total_lines        
    stats["avg_lines"] = total_lines/len(articles)

    total_headlines = 0
    total_headline_words = 0
    for i,headline in enumerate(tqdm.tqdm(headlines)):
        headlines[i] = text_processor.run(headline)
        lang = model.predict([line])[0][0][0].split("__")[-1]
        num_words = len(headlines[i].split())
        total_headline_words += num_words
        total_headlines += 1

        if lang == "en":
            stats["en_headlines"] += 1
        elif lang == "hi":
            if TRANSLATE:
                headlines[i] = translate(headlines[i]) 
            stats["hi_headlines"] += 1
        else:
            if TRANSLATE:
                headlines[i] = translate(headlines[i]) 
            stats["eh_headlines"] += 1

        stats["max_headline_len"] = max(stats["max_headline_len"], num_words)
        stats["min_headline_len"] = min(stats["min_headline_len"], num_words)

    stats["total_headlines"] = total_headlines
    stats["avg_headline_len"] = total_headline_words/total_headlines
    pprint.pprint(stats)
    json.dump(stats, open(STATS_PATH, "w"))

    proc = []
    for headline, article in zip(headlines, articles):
        proc.append({"text":article, "summary":headline})
    
    train, test = train_test_split(proc, test_size=0.1)
    test = pd.DataFrame(test)

    train = pd.DataFrame(train)
    train_path = os.path.join(PROCESSED_PATH, "train.csv")
    test_path = os.path.join(PROCESSED_PATH, "test.csv")
    # print(train, test)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

# if TRANSLATE:
#     index = 0
#     def translate(text):
#         return text

#     for headline, article in zip(headlines, articles):
#         trans_dataset[index]['en_Headline'] = translate(headline)
#         trans_dataset[index]['en_Text'] = translate(article)
#         index += 1