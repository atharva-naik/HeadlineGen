# first create a structured json with language identification
import os
import json
import tqdm
import fasttext
import numpy as np
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


def augmentCSV(path, keep=['text', 'summary'], article_field='text'):
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

        article[article_field] = temp
        augmented.append(article)

    json.dump(augmented, open(path.replace(
        ".csv", ".json").replace(".xlsx", ".json"), "w"), indent=4)


if __name__ == "__main__":
    augmentCSV(TRAIN)
    augmentCSV(TEST)
    augmentCSV(VAL)
