import torch
import json
import pickle
from tqdm import tqdm
import sys
import numpy as np
l = pickle.load(open("em_E.pk_all_1.0", "rb"))
n = 0
rst = []
thres = 0.9
for d in tqdm(l):
    grain_size = d['grain_size']
    ori_relations = d['ori_relations'][:,2]
    relations = d['relations'][:,2]
    for i, (grain, ori_rels, rels) in enumerate(zip(grain_size, ori_relations, relations)):
        if grain < thres:
            # do not transfer
            d['relations'][i][2] = 0
        if (1 - grain) >= thres:
            d['relations'][i][2] = ori_rels
    rst.append(d)
# pickle.dump(rst, open("em_E.pk"+str(round(ratio, 2)), "wb"))
pickle.dump(rst, open("em_E_aove.pk", "wb"))
print(n)
