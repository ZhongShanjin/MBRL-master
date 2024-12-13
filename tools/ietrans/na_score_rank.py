import torch
import json
import pickle
from tqdm import tqdm
l = pickle.load(open("ex_confusion_ori.pk", "rb"))

def score():
    rst = []
    correct = 0
    n = 0
    for d in tqdm(l):
        logits = d['rel_logits']
        for lg in logits:
            rst.append(torch.tensor(lg).softmax(0)[-1].item())
    return rst

rst = score()
rst = sorted(rst)
json.dump(rst, open("score.json", "w"))

