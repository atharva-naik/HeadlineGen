import os 
import re 
import copy
import json
import pandas as pd 
# try: 
#     import google_trans_new
# except ImportError:
#     os.system("pip install google_trans_new")
# try: 
#     import emoji
# except ImportError:
#     os.system("pip install emoji")

for lib in ["emoji", "fasttext", "google_trans_new"]:
    try: 
        exec(f"import {lib}")
    except ImportError:
        os.system(f"pip install {lib}")

DATASET_PATH = "raw_data/Development Data/dev_data_article.xlsx"
TRANSLATED_PATH = "raw_data/Translated Data/"
TRANSLATE = False
STATS_PATH = "stats.json"

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
    def __init__(self, filts):
        self.filts = filts 
        self.valid_filts = VALID_FILTERS
        
    def add(self, filt):
        self.filts.append(filt)

    @classmethod
    def Sequential(cls, filts):
        cls.filts = []
        cls.valid_filts = VALID_FILTERS
        for filt in filts:
            cls.filts.append(cls.valid_filts.get(filt, identity))

    def run(self, text):
        for filt in self.filts:
            text = filt(text)

        return text 

text_processor = ProcessText.Sequential(["strip",
                                     "remove_urls", 
                                     "remove_punctuation"])
stats = {"max_lines":0,
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

STATS_ABSENT = True
try:
    open(STATS_PATH)
except FileNotFoundError:
    STATS_ABSENT = True
    try:
        open('/home/atharva/lid.176.bin', "rb")
    except FileNotFoundError:
        os.system("wget -O /home/atharva/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
    PRETRAINED_MODEL_PATH = '/home/atharva/lid.176.bin'
    model = fasttext.load_model(PRETRAINED_MODEL_PATH)

if STATS_ABSENT:
    total_lines = 0
    total_words = 0

    for article in articles:
        for i,para in enumerate(articles.split("\n")):
            article_lines = 0
            proc_article = ''
            for line in para.split("."):
                line = line.strip()
                if line == "":
                    total_lines += 1
                    article_lines += 1
                    proc_article += text_processor.run(line)+'\n'
                    
                    lang = model.predict([line])[0][0][0].split("__")[-1]
                    if lang == 'hi':
                        stats["hi_lines"] += 1
                    elif lang == 'en':
                        stats["en_lines"] += 1
                    else:
                        stats["eh_lines"] += 1

                    num_words = len(line.split())
                    total_words += num_words
                    stats["max_line_len"] = max(num_words, stats["max_line_len"])
                    stats["min_line_len"] = min(num_words, stats["min_line_len"])

            stats["max_lines"] = max(article_lines, stats["max_lines"])
            stats["min_lines"] = min(article_lines, stats["min_lines"])
            proc_article.strip().strip("\n")
            articles[i] = proc_article

        stats["avg_line_len"] = total_words/total_lines        
        stats["avg_lines"] = total_lines/len(articles)

    for i,headline in enumerate(headlines):
        headlines[i] = text_processor.run(headline)
    json.dump(stats, open(STATS_PATH, "w"))

if TRANSLATE:
    index = 0
    def translate(text):
        return text

    for headline, article in zip(headlines, articles):
        trans_dataset[index]['en_Headline'] = translate(headline)
        trans_dataset[index]['en_Text'] = translate(article)
        index += 1