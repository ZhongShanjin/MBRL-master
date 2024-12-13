import json
import pickle
import sys
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
import IPython

path = "./hlm/motifs.pk_HLM_1.0"

vocab = json.load(open("VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
lb2idx = {k: int(v) for k, v in vocab["label_to_idx"].items()}
idx2pred = {int(k)-1: v for k, v in vocab["idx_to_predicate"].items()}
pred2idx = {k: int(v)-1 for k, v in vocab["predicate_to_idx"].items()}

l = pickle.load(open(path, "rb"))

for i, data in tqdm(enumerate(l)):
    labels = data["labels"]
    logits = data["logits"][:, 1:]
    relation_tuple = deepcopy(data["relations"])
    ori_relation_tuple = deepcopy(data["ori_relations"])
    sub_idxs, obj_idxs, rels = relation_tuple[:, 0], relation_tuple[:, 1], relation_tuple[:, 2]
    sub_lbs, obj_lbs = labels[sub_idxs], labels[obj_idxs]
    # [[sub_lb1, obj_lb1], [sub_lb2, obj_lb2]......]
    pairs = np.stack([sub_lbs, obj_lbs], 1).tolist()
    pairs = [(idx2lb[p[0]], idx2lb[p[1]]) for p in pairs]
    ori_rels = ori_relation_tuple[:, 2] - 1
    # behave as indexes, so -=1
    rels -= 1
    # fill in rel_dic
    # rel_dic: {rel_i: {pair_j: distribution} }
    for j, (pair, r, logit, orir) in enumerate(zip(pairs, rels, logits, ori_rels)):
        r_name = idx2pred[int(r)]
        orir_r_name = idx2pred[int(orir)]
        # orir_r_name!="sitting on"
        if (pair[0]!="plane" or pair[1]!="plane" or r_name!="flying in"):
            continue
        end = 'end'

pickle.dump(l, open("matrix.pk"), "wb")


