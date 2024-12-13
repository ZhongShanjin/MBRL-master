import json
import pickle
import sys
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
import IPython
path = str(sys.argv[1])
save_path = str(sys.argv[2])
print(path)
print(save_path)

vocab = json.load(open("VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
lb2idx = {k: int(v) for k, v in vocab["label_to_idx"].items()}
idx2pred = {int(k)-1: v for k, v in vocab["idx_to_predicate"].items()}
pred2idx = {k: int(v)-1 for k, v in vocab["predicate_to_idx"].items()}

l = pickle.load(open(path, "rb"))

all_triplet_idxs = {}
all_triplet_subs = []
all_triplet_pred_subs = []
all_triplet_objs = []
all_triplet_pred_objs = []
all_triplet_rels = []
all_triplet_rels = []
all_triplet_logits = []
all_triplet_pred = []

n = 0
for i, data in tqdm(enumerate(l)):
    labels = data["labels"]
    logits = data["logits"][:, 1:]
    lable_logits = data["label_logits"]
    pred_labels = np.argmax(lable_logits, 1)
    relation_tuple = deepcopy(data["relations"])
    sub_idxs, obj_idxs, rels = relation_tuple[:, 0], relation_tuple[:, 1], relation_tuple[:, 2]
    sub_lbs, obj_lbs = labels[sub_idxs], labels[obj_idxs]
    pred_sub_lbs, pred_obj_lbs = pred_labels[sub_idxs], pred_labels[obj_idxs]
    # [[sub_lb1, obj_lb1], [sub_lb2, obj_lb2]......]
    pairs = np.stack([sub_lbs, obj_lbs], 1).tolist()
    pred_pairs = np.stack([pred_sub_lbs, pred_obj_lbs], 1).tolist()
    pairs = [(idx2lb[p[0]], idx2lb[p[1]]) for p in pairs]
    pred_pairs = [(idx2lb[p[0]], idx2lb[p[1]]) for p in pred_pairs]
    # behave as indexes, so -=1
    rels -= 1

    for j, (pair, r, logit) in enumerate(zip(pairs, rels, logits)):
        logit = torch.softmax(torch.from_numpy(logit), 0)

        all_triplet_rels.append(r)
        all_triplet_subs.append(lb2idx[pair[0]])
        all_triplet_pred_subs.append(lb2idx[pred_pairs[j][0]])
        all_triplet_pred_objs.append(lb2idx[pred_pairs[j][1]])
        all_triplet_objs.append(lb2idx[pair[1]])
        all_triplet_logits.append(logit)
        all_triplet_pred.append(torch.argmax(logit))
        all_triplet_idxs[n] = (i, j)
        n += 1

all_triplet_rels = np.asarray(all_triplet_rels)
all_triplet_subs = np.asarray(all_triplet_subs)
all_triplet_pred_subs = np.asarray(all_triplet_pred_subs)
all_triplet_objs = np.asarray(all_triplet_objs)
all_triplet_pred_objs = np.asarray(all_triplet_pred_objs)
all_triplet_pred = np.asarray(all_triplet_pred)
all_triplet_logits = torch.stack(all_triplet_logits, 0)
print(len(all_triplet_rels), len(all_triplet_subs), len(all_triplet_objs), all_triplet_logits.size(), all_triplet_pred.size, all_triplet_pred_objs.size, all_triplet_pred_subs.size)
assert len(all_triplet_rels) == len(all_triplet_subs) == len(all_triplet_objs) == len(all_triplet_logits)
assert n == len(all_triplet_rels)
assert len(all_triplet_idxs) == len(all_triplet_rels)

confusion_dic = {}
for fus_id, fus_pred in tqdm(enumerate(all_triplet_pred)):
    pred_sub_lb, pred_obj_lb = all_triplet_pred_subs[fus_id], all_triplet_pred_objs[fus_id]
    pair_flag = (all_triplet_pred_subs == pred_sub_lb) & (all_triplet_pred_objs == pred_obj_lb)
    rels_flag = np.zeros_like(pair_flag) != 0
    rels_flag |= (all_triplet_pred == fus_pred)
    flag = rels_flag & pair_flag
    gt_rels = all_triplet_rels[flag]
    sub_lb, obj_lb = all_triplet_subs[flag], all_triplet_objs[flag]
    r_name = (pred_sub_lb, pred_obj_lb, fus_pred + 1)

    if r_name not in confusion_dic:
        confusion_dic[r_name] = {}
        confusion_dic[r_name]['rel'] = torch.zeros(50)
        confusion_dic[r_name]['sub'] = torch.zeros(151)
        confusion_dic[r_name]['obj'] = torch.zeros(151)
        for gt_rel, gt_sub, gt_obj in zip(gt_rels, sub_lb, obj_lb):
            confusion_dic[r_name]['rel'][gt_rel] += 1
            confusion_dic[r_name]['sub'][gt_sub] += 1
            confusion_dic[r_name]['obj'][gt_obj] += 1
        confusion_dic[r_name]['rel'] = torch.softmax(confusion_dic[r_name]['rel'], 0)
        confusion_dic[r_name]['sub'] = torch.softmax(confusion_dic[r_name]['sub'], 0)
        confusion_dic[r_name]['obj'] = torch.softmax(confusion_dic[r_name]['obj'], 0)
        end = 'end'

pickle.dump(confusion_dic, open(save_path, "wb"))
end='end'

