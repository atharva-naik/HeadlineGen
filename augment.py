# first create a structured json with language identification
import os
import json
import tqdm
import fasttext
import numpy as np
import pandas as pd
from google_trans_new import google_translator
# this code dumps model at your home directory
HOME = os.getenv("HOME")
RAW_PATH = "raw_data/Development Data/dev_data_article.xlsx"
MODEL_PATH = os.path.join(HOME, "lid.176.bin")
GOOGLE_TRANS_ENGINE = google_translator()
TRAIN = "proc_data/train.csv"
TEST = "proc_data/test.csv"
VAL = "proc_data/val.csv"

try:
    open(MODEL_PATH, "rb")
except FileNotFoundError:
    os.system(
        f"wget -O {MODEL_PATH} https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
model = fasttext.load_model(MODEL_PATH)


def batchLangID(batch, filter_lang=False, lang='en', thresh=0.9):
    preds = model.predict(batch)
    langs = np.concatenate(preds[0])
    conf = np.concatenate(preds[1])
    bitmask = np.ones(len(langs))

    if filter_lang:
        lang_bitmask = np.array([lang == '__lang__en' for lang in langs])
        thresh_bitmask = np.array([c >= thresh for c in conf])
        bitmask = lang_bitmask * thresh_bitmask

    return langs, conf, bitmask


def batchTranslate(batch, mask=None, dest='hi'):
    trans_text = []
    for i, text in enumerate(batch):
        if mask and mask[i] == True:
            trans_text.append(GOOGLE_TRANS_ENGINE.translate(text, dest='hi'))
        elif mask and mask[i] == False:
            trans_text.append(trans_text)
        else:
            trans_text.append(GOOGLE_TRANS_ENGINE.translate(text, dest='hi'))

    return trans_text


def augmentCSV(path, keep=['text', 'summary'], article_field='text', summary_field='summary'):
    import pandas as pd

    if path.endswith(".csv"):
        dataset = pd.read_csv(path)
    elif path.endswith(".xlsx"):
        dataset = pd.read_excel(path)
    else:
        raise(TypeError)
    augmented = []

    for article in tqdm.tqdm(dataset[keep].to_dict('records')):
        text = article[article_field]
        temp = []

        lines = text.split("\n")
        langs, confs, _ = batchLangID(lines)
        for line, lang, conf in zip(lines, langs, confs):
            temp.append({"text": " ".join(line.split()), "lang": lang, "conf": str(conf)})

        article[summary_field] = " ".join(article[summary_field].split())
        article[article_field] = temp
        augmented.append(article)

    json.dump(augmented, open(path.replace(".csv", ".json").replace(".xlsx", ".json"), "w"), indent=4)

def getAllBitSrings(n: int):
    import math
    setB = []
    n = 2**n-1

    length = len(bin(n)[2:])

    for i in range(n+1):
        setB.append([])
        for j in bin(i)[2:].rjust(length,'0'):
            setB[-1].append(int(j))

    return setB

def getMasks(n: int, map_: dict):
    helper_mask = getAllBitSrings(len(map_))
    masks = []
    # print(helper_mask)

    for mask in helper_mask:
        masks.append([0 for i in range(n)])
        for j,val in enumerate(mask):
            if val == 1:
                # print(n)
                # print(map_[j])
                masks[-1][map_[j]] = 1        

    return masks

def augment(path, thresh):
    import copy
    datatset = json.load(open(path))
    num_lines = 0
    new_articles = [] 
    new_num_articles = 0

    for article in tqdm.tqdm(datatset):
        index_map = {}
        j = 0
        for i,line in enumerate(article["text"]):
            if line['lang'] == "__label__en"  and float(line['conf']) >= thresh and j<=5:
                index_map[j] = i
                num_lines += 1
                j += 1
        # print(len(index_map))
        new_num_articles += 2**len(index_map)

        for mask in getMasks(len(article["text"]), index_map):
            lines = []

            for idx, val in enumerate(mask):
                if val == 1:
                    try:
                        # print(idx, index_map[idx])
                        lines.append(article['text'][idx]['hi_text'])
                        # print(article['text'][idx]['hi_text'])
                    except KeyError:
                        lines.append(article['text'][idx]['text'])
                else:
                    lines.append(article['text'][idx]['text'])
            new_articles.append({"text":"\n".join(lines), "summary":article["summary"]})
    
    print(f"predicted growth factor = {2**(num_lines/len(datatset))}")    
    print(f"actual growth factor = {new_num_articles/len(datatset)}")  
    print(len(new_articles))
    # print(len(new_articles))
    pd.DataFrame(new_articles).to_csv("proc_data/train_augmented.csv", index=False)
    

if __name__ == "__main__":
    # augmentCSV(TRAIN)
    # augmentCSV(TEST)
    # augmentCSV(VAL)
    augment("proc_data/train_augmented_article_208_done.json", 0.97) 