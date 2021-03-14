import json 
import fasttext
import pandas as pd
import matplotlib.pyplot as plt 

TEST_JSON = "proc_data/test.json"
TEST_CSV = "proc_data/test.csv"
SUMMARIES1 = "summaries/hindi-and-english-headlines.txt"
SUMMARIES2 = "summaries/english-only-headlines.txt"

test_dataset = json.load(open(TEST_JSON))
test_csv = pd.read_csv(TEST_CSV).to_dict("records")
summaries1 = open(SUMMARIES1).read().split("\n")
summaries2 = open(SUMMARIES2).read().split("\n")

HINGLISH_INDICES = []
english_dist = [0 for i in range(1000)]
hindi_dist =[0 for i in range(1000)]
hinglish_dist = [0 for i in range(101)] 
for i,test in enumerate(test_dataset):
    num_en = 0
    num_hi = 0
    num_eh = 0

    for line in test["text"]:
        if line["lang"] == "__label__en":
            if float(line["conf"]) <= 0.65:
                num_eh += 1
            else:
                num_en += 1
        elif line["lang"] == "__label__hi": 
            num_hi += 1   

    # labels = [num_en, num_hi, num_eh]
    # label = labels.index(max(labels))
    # if label == 2:
    #     HINGLISH_INDICES.append(i)
    hinglish_dist[min(num_eh, 100)] += 1
    if num_eh >= 1:
        HINGLISH_INDICES.append(i)

summaries_dataset = []
for i in HINGLISH_INDICES:
    sum1 = summaries1[i]
    sum2 = summaries2[i]
    article = test_csv[i]["text"]
    summaries_dataset.append({"article":article, "summary1":sum1, "summary2":sum2})

import copy
hinglish_dist_cum = copy.deepcopy(hinglish_dist)
for i in range(len(hinglish_dist)-1):
    hinglish_dist_cum[i+1] += hinglish_dist_cum[i]

print(hinglish_dist_cum)
hinglish_dist_cum = hinglish_dist_cum[0:10]
print(len(summaries_dataset))
pd.DataFrame(summaries_dataset).to_csv("hinglish_summaries_expanded.csv", index=False)

x = [i for i,_ in enumerate(hinglish_dist_cum)]
y = hinglish_dist_cum
plt.figure(figsize=(15,9))
plt.plot(x, y)

for a,b in zip(x, y): 
    plt.text(a, b, f"({a},{b})")
plt.xticks(x, labels=x)
plt.yticks([50*i for i in range(11)], labels=[50*i for i in range(11)])

plt.title("Hinglish sentence distribution for test set")
plt.xlabel("number of hinglih sentences (x)")
plt.ylabel("number of articles with more than x hinglish sentences")
plt.savefig("plots/hinglish_dist.png")