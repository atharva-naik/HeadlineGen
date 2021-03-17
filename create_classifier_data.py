import tqdm
import scipy
import pandas as pd 
from preprocess import process_dataset
from sentence_transformers import SentenceTransformer

similarity_model = SentenceTransformer('bert-base-nli-mean-tokens')

print("preprocessing article dataset")
pipeline = ["strip", "remove_emoji", "remove_newline", "remove_url", "remove_punctuation", "insert_newline", "lower"]
train, val, test = process_dataset("all_english_headlines.csv", pipeline, 69, valid_size=1/8, test_size=1/8)

train_data_classifier = []
test_data_classifier = []
val_data_classifier = []

print("generating classifier dataset")
for record in tqdm.tqdm(train.to_dict('records')):
    for line in record["text"].split("."):
        true = similarity_model.encode(record["summary"])
        pred = similarity_model.encode(line)
        
        similarity = 1-scipy.spatial.distance.cdist([true], [pred], "cosine")
        train_data_classifier.append({"line":line,"headline":record["summary"],"score":similarity})   
pd.DataFrame(train_data_classifier).to_csv("proc_data/train_classifier.csv", index=False)

for record in tqdm.tqdm(test.to_dict('records')):
    for line in record["text"].split("."):
        true = similarity_model.encode(record["summary"])
        pred = similarity_model.encode(line)
        
        similarity = 1-scipy.spatial.distance.cdist([true], [pred], "cosine")
        train_data_classifier.append({"line":line,"headline":record["summary"],"score":similarity})  
pd.DataFrame(test_data_classifier).to_csv("proc_data/test_classifier.csv", index=False)

for record in tqdm.tqdm(val.to_dict('records')):
    for line in record["text"].split("."):
        true = similarity_model.encode(record["summary"])
        pred = similarity_model.encode(line)
        
        similarity = 1-scipy.spatial.distance.cdist([true], [pred], "cosine")
        train_data_classifier.append({"line":line,"headline":record["summary"],"score":similarity})  
pd.DataFrame(val_data_classifier).to_csv("proc_data/val_classifier.csv", index=False)
        