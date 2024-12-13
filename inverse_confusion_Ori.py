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
# path = "em_ori_confusion.pk"

vocab = json.load(open("VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
lb2idx = {k: int(v) for k, v in vocab["label_to_idx"].items()}
idx2pred = {int(k)-1: v for k, v in vocab["idx_to_predicate"].items()}
pred2idx = {k: int(v)-1 for k, v in vocab["predicate_to_idx"].items()}

l = pickle.load(open(path, "rb"))

all_triplet_idxs = {}
all_triplet_subs = []
all_triplet_objs = []
all_triplet_rels = []
all_triplet_logits = []

n = 0
for i, data in tqdm(enumerate(l)):
    labels = data["labels"]
    logits = data["logits"][:, 1:]
    relation_tuple = deepcopy(data["relations"])
    sub_idxs, obj_idxs, rels = relation_tuple[:, 0], relation_tuple[:, 1], relation_tuple[:, 2]
    sub_lbs, obj_lbs = labels[sub_idxs], labels[obj_idxs]
    # [[sub_lb1, obj_lb1], [sub_lb2, obj_lb2]......]
    pairs = np.stack([sub_lbs, obj_lbs], 1).tolist()
    pairs = [(idx2lb[p[0]], idx2lb[p[1]]) for p in pairs]
    # behave as indexes, so -=1
    rels -= 1

    for j, (pair, r, logit) in enumerate(zip(pairs, rels, logits)):

        logit = torch.softmax(torch.from_numpy(logit), 0)

        all_triplet_rels.append(r)
        all_triplet_subs.append(lb2idx[pair[0]])
        all_triplet_objs.append(lb2idx[pair[1]])
        all_triplet_logits.append(logit)
        all_triplet_idxs[n] = (i, j)
        n += 1

all_triplet_rels = np.asarray(all_triplet_rels)
all_triplet_subs = np.asarray(all_triplet_subs)
all_triplet_objs = np.asarray(all_triplet_objs)
all_triplet_logits = torch.stack(all_triplet_logits, 0)
print(len(all_triplet_rels), len(all_triplet_subs), len(all_triplet_objs), all_triplet_logits.size())
assert len(all_triplet_rels) == len(all_triplet_subs) == len(all_triplet_objs) == len(all_triplet_logits)
assert n == len(all_triplet_rels)
assert len(all_triplet_idxs) == len(all_triplet_rels)

confusion_dic = {}
for fus_id, gt_rel in tqdm(enumerate(all_triplet_rels)):
    sub_lb, obj_lb = all_triplet_subs[fus_id], all_triplet_objs[fus_id]
    confusion_pairs = (sub_lb, obj_lb)

    if confusion_pairs not in confusion_dic:
        confusion_dic[confusion_pairs] = {}
    if gt_rel not in confusion_dic[confusion_pairs]:
        pair_flag = (all_triplet_subs == sub_lb) & (all_triplet_objs == obj_lb)
        rels_flag = np.zeros_like(pair_flag) != 0
        rels_flag |= (all_triplet_rels == gt_rel)
        flag = rels_flag & pair_flag
        rel_logits = all_triplet_logits[flag]
        confusion_dic[confusion_pairs][gt_rel] = torch.zeros(50)
        for pred_logit in rel_logits:
            the_fus_rel = torch.where(pred_logit > pred_logit[gt_rel])[0]
            the_max_rel = torch.argmax(pred_logit)
            if the_max_rel == gt_rel:
                confusion_dic[confusion_pairs][gt_rel][gt_rel] += 1
            else:
                confusion_dic[confusion_pairs][gt_rel][the_fus_rel] += 1
            end = 'end'

confusion_pairs_dic = {}
for pair_id, fus_pairs in tqdm(enumerate(confusion_dic)):
    confusion_pairs_dic[fus_pairs] = {}
    for i in range(50):
        confusion_pairs_dic[fus_pairs][i] = torch.zeros(50)
    for fusion_rel in confusion_dic[fus_pairs]:
        for i in range(50):
            confusion_pairs_dic[fus_pairs][i][fusion_rel] += confusion_dic[fus_pairs][fusion_rel][i]
    for fusion_rel in confusion_dic[fus_pairs]:
        for i in range(50):
            all_value = deepcopy(torch.sum(confusion_pairs_dic[fus_pairs][i]))
            if all_value > 0:
                for j in range(50):
                    confusion_pairs_dic[fus_pairs][i][j] = confusion_pairs_dic[fus_pairs][i][j] / all_value
    end = 'end'

# pickle.dump(confusion_pairs_dic, open("ori_inverse_confusion.pk", "wb"))
pickle.dump(confusion_pairs_dic, open(save_path, "wb"))
end='end'

