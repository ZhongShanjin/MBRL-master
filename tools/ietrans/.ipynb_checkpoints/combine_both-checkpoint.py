import json, pickle
import numpy as np
import sys, os
from tqdm import tqdm
import torch
from copy import deepcopy
import IPython
# extra: ['img_path', 'boxes', 'labels', 'pairs', 'possible_rels', 'rel_logits']
# intra: ['width', 'height', 'img_path', 'boxes', 'labels', 'relations', 'possible_rels', 'logits']

topk_percent = float(1)
sgcls_ori_path = "../checkpoint/motif-sgcls/sgcls_confusion_in.pk"
ori_path = "em_E.pk"
extra_path = "em_E_aove.pk"
intra_path = "em_E.pk_all_1.0"
print("extra_path", extra_path)
print("intra_path", intra_path)
print("ori_path", ori_path)
img_info_path = "datasets/vg/image_data.json"
extra_data = pickle.load(open(extra_path, "rb"))
intra_data = pickle.load(open(intra_path, "rb"))
ori_data = pickle.load(open(ori_path, "rb"))
sgcls_ori_data = pickle.load(open(sgcls_ori_path, "rb"))
img_infos = json.load(open(img_info_path, "r"))
img_infos = {k["image_id"]: k for k in img_infos}

vocab = json.load(open("VG-SGG-dicts-with-attri.json", "r"))
idx2lb = {int(k): v for k, v in vocab["idx_to_label"].items()}
lb2idx = {k: int(v) for k, v in vocab["label_to_idx"].items()}
idx2pred = {int(k)-1: v for k, v in vocab["idx_to_predicate"].items()}
pred2idx = {k: int(v)-1 for k, v in vocab["predicate_to_idx"].items()}
rel_dic = {}
rel_cnt_dic = {}
pair_dic = {}
all_triplet_idxs = {}
all_triplet_subs = []
all_triplet_objs = []
all_triplet_rels = []
all_triplet_logits = []
all_triplet_sub_logits = []
all_triplet_obj_logits = []
need_triplet = []
n = 0
for i, data in tqdm(enumerate(sgcls_ori_data)):
    labels = data["labels"]
    logits = data["logits"][:, 1:]
    label_logits = data["label_logits"]
    relation_tuple = deepcopy(data["ori_relations"])
    sub_idxs, obj_idxs, rels = relation_tuple[:, 0], relation_tuple[:, 1], relation_tuple[:, 2]
    sub_lbs, obj_lbs = labels[sub_idxs], labels[obj_idxs]
    # [[sub_lb1, obj_lb1], [sub_lb2, obj_lb2]......]
    pairs = np.stack([sub_lbs, obj_lbs], 1).tolist()
    pairs = [(idx2lb[p[0]], idx2lb[p[1]]) for p in pairs]
    # behave as indexes, so -=1
    rels -= 1

    for j, (pair, r, logit) in enumerate(zip(pairs, rels, logits)):
        r_name = idx2pred[int(r)]

        if r_name not in rel_cnt_dic:
            rel_cnt_dic[r_name] = {}
        if pair not in rel_cnt_dic[r_name]:
            rel_cnt_dic[r_name][pair] = 0
        rel_cnt_dic[r_name][pair] += 1

        logit = torch.softmax(torch.from_numpy(logit), 0)
        if r_name not in rel_dic:
            rel_dic[r_name] = {}
        if pair not in rel_dic[r_name]:
            rel_dic[r_name][pair] = 0
        rel_dic[r_name][pair] += logit

        if r_name not in pair_dic:
            pair_dic[r_name] = {}
        if pair not in pair_dic[r_name]:
            pair_dic[r_name][pair] = {}
            pair_dic[r_name][pair]['sub'] = 0
            pair_dic[r_name][pair]['obj'] = 0
        pair_dic[r_name][pair]['sub'] += label_logits[sub_idxs[j]]
        pair_dic[r_name][pair]['obj'] += label_logits[obj_idxs[j]]

        all_triplet_rels.append(r)
        all_triplet_subs.append(lb2idx[pair[0]])
        all_triplet_objs.append(lb2idx[pair[1]])
        all_triplet_logits.append(logit)
        all_triplet_sub_logits.append(label_logits[sub_idxs[j]])
        all_triplet_obj_logits.append(label_logits[obj_idxs[j]])
        all_triplet_idxs[n] = (i, j)
        n += 1
all_triplet_rels = np.asarray(all_triplet_rels)
all_triplet_subs = np.asarray(all_triplet_subs)
all_triplet_objs = np.asarray(all_triplet_objs)
all_triplet_logits = torch.stack(all_triplet_logits, 0)
all_triplet_sub_logits = np.stack(all_triplet_sub_logits, 0)
all_triplet_obj_logits = np.stack(all_triplet_obj_logits, 0)
print(len(all_triplet_rels), len(all_triplet_subs), len(all_triplet_objs), all_triplet_logits.size())
assert len(all_triplet_rels) == len(all_triplet_subs) == len(all_triplet_objs) == len(all_triplet_logits) == len(all_triplet_sub_logits) == len(all_triplet_obj_logits)
assert n == len(all_triplet_rels)
assert len(all_triplet_idxs) == len(all_triplet_rels)
all_changes = np.zeros_like(all_triplet_rels, dtype=np.float)

def vis_triplet(triplet):
    logit = rel_dic[triplet[0]][(triplet[1], triplet[2])]
    sub_logit = pair_dic[triplet[0]][(triplet[1], triplet[2])]['sub']
    obj_logit = pair_dic[triplet[0]][(triplet[1], triplet[2])]['obj']
    scores, idxs = logit.sort(descending=True)
    sub_scores, sub_idxs = torch.from_numpy(sub_logit).sort(descending=True)
    obj_scores, obj_idxs = torch.from_numpy(obj_logit).sort(descending=True)
    prds = [idx2pred.get(int(i), "none") for i in idxs]
    sub_prds = [idx2lb.get(int(i+1), "none") for i in sub_idxs]
    obj_prds = [idx2lb.get(int(i+1), "none") for i in obj_idxs]
    return list(zip(prds, scores)), list(zip(sub_prds, sub_scores)), list(zip(obj_prds, obj_scores))

def vis_pair_triplet(triplet):
    logit = rel_dic[triplet[0]][(triplet[1], triplet[2])]
    scores, idxs = logit.sort(descending=True)
    prds = [idx2pred.get(int(i), "none") for i in idxs]
    return list(zip(prds, scores))

def collect_all_parent_data(query_parents, query_pair, son_rel_idx):
    sub_lb, obj_lb = lb2idx[query_pair[0]], lb2idx[query_pair[1]]
    pair_flag = (all_triplet_subs==sub_lb) & (all_triplet_objs==obj_lb)
    rel_flag = np.zeros_like(pair_flag) != 0
    for p in query_parents:
        qp_idx = pred2idx[p]
        rel_flag |= (all_triplet_rels == qp_idx)
    flag = rel_flag & pair_flag
    logits = all_triplet_logits[flag, son_rel_idx]
    return np.where(flag)[0], logits

importance_dic = {}
for r, pair_cnt_dic in tqdm(rel_cnt_dic.items()):
    for pair in pair_cnt_dic:
        cnt = pair_cnt_dic[pair]
        triplet = (r, *pair)
        importance_dic[triplet] = cnt/sum(pair_cnt_dic.values())

all_triplets = []
for r, pair_cnt_dic in tqdm(rel_cnt_dic.items()):
    for pair in pair_cnt_dic:
        all_triplets.append((r, *pair))

for triplet in tqdm(all_triplets):
    # IPython.embed()
    # triplet: (r, sub, obj)
    r = triplet[0]
    sub, obj = triplet[1], triplet[2]
    # prds: [(rel, score)]
    prds, sub_prds, obj_prds = vis_triplet(triplet)
    # find parents
    parents = [p[0] for p in prds]
    sub_parents = [p[0] for p in sub_prds]
    obj_parents = [p[0] for p in obj_prds]
    parents = parents[: parents.index(r) + 1]
    sub_parents = sub_parents[: sub_parents.index(sub) + 1]
    obj_parents = obj_parents[: obj_parents.index(obj) + 1]
    # filter parents:
    # if current triplet is more important for node, the node is a son
    # parents = [ p for p in parents if importance_dic.get((p, triplet[1], triplet[2]), 0) < importance_dic[triplet] ]
    parents = [ p for p in parents ]
    sub_parents = [ p for p in sub_parents ]
    obj_parents = [ p for p in obj_parents ]

    if importance_dic[triplet] > importance_dic.get((parents[0], sub_parents[0], obj_parents[0]), 0):
        need_triplet.append(triplet)

def get_img_id(path):
    return int(path.split("/")[-1].replace(".jpg", ""))

def to_dic(data):
    dic = {}
    for d in tqdm(data):
        dic[get_img_id(d['img_path'])] = d
    return dic

def complete_img_info(data, img_infos):
    imid = get_img_id(data["img_path"])
    iminfo = img_infos[imid]
    data["width"] = iminfo["width"]
    data["height"] = iminfo["height"]

def filter_out_freq_rels(relations):
    rels = relations[:, 2]
    in_mask = (rels.reshape(-1, 1) == freq_rels.reshape(-1, 1).T).any(-1)
    return relations[~in_mask]

def complete_relatinos(data):
    assert "rel_logits" in data, data
    rel_logits = data["rel_logits"]
    possible_rels = data["possible_rels"]
    pairs = data["pairs"]
    rels = []
    for poss_rls, logits, pair in zip(possible_rels, rel_logits, pairs):
        max_id = logits.argmax()
        sub_pair, obj_pair = pair
        if (idx2pred.get(poss_rls[max_id]-1,'none'), idx2lb.get(data['labels'][sub_pair],'none'), idx2lb.get(data['labels'][obj_pair],'none')) in need_triplet:
            rels.append(poss_rls[max_id])
        else:
            rels.append(int(0))
    rels = np.array(rels, dtype=pairs.dtype).reshape(-1, 1)
    rels = np.concatenate([pairs, rels], 1)
    data["relations"] = filter_out_freq_rels(rels)
    del data["pairs"]
    return data

def ex_data_to_in_data(ex_data, img_infos):
    complete_img_info(ex_data, img_infos)
    complete_relatinos(ex_data)
    del ex_data["rel_logits"]
    return ex_data

def merge_ex_and_in(ex_data, in_data):
    assert ex_data["img_path"] == in_data["img_path"]
    ex_data = ex_data_to_in_data(ex_data, img_infos)
    ex_rels = ex_data["relations"]
    ex_pairs = ex_rels[:, :2]
    in_rels = in_data["relations"]
    in_pairs = in_rels[:, :2]
    intersect_rels = ex_pairs.reshape(ex_pairs.shape[0], ex_pairs.shape[1], 1) \
                     == in_pairs.reshape(in_pairs.shape[0], in_pairs.shape[1], 1).T
    intersect_rels = intersect_rels.all(-1).any(1)
    unintersect_rels = ~intersect_rels
    final_rels = np.concatenate([in_rels, ex_rels[unintersect_rels]], 0)
    in_data["relations"] = final_rels
    return in_data

print("intra", len(intra_data), sum([len(x["relations"]) for x in intra_data]) )

# to dic
# extra: ['img_path', 'boxes', 'labels', 'pairs', 'possible_rels', 'rel_logits']
# intra: ['width', 'height', 'img_path', 'boxes', 'labels', 'relations', 'possible_rels', 'logits']
extra_data = to_dic(extra_data)
intra_data = to_dic(intra_data)
to_save_data = {}
# filter_out relations in extra
freq_rels = [0, 31, 20, 22, 30, 48, 29, 50, 1, 21, 8, 43, 40, 49, 41, 23]
# freq_rels = [0]
# freq_rels = {r: 0 for r in freq_rels}
freq_rels = np.array(freq_rels, dtype=np.int64)
for i, (img_id, ex_data) in tqdm(enumerate(extra_data.items()), total=len(extra_data)):
    in_data = intra_data.get(img_id, None)

    if not in_data:
        tmp = ex_data_to_in_data(ex_data, img_infos)
        if len(tmp["relations"]) == 0:
            continue
        to_save_data[img_id] = tmp

    if in_data:
        to_save_data[img_id] = merge_ex_and_in(ex_data, in_data)
to_save_data = list(to_save_data.values())
pickle.dump(to_save_data, open("em_EE.pk", "wb"))

# stat relations
print("saved data:", len(to_save_data), sum([len(x["relations"]) for x in to_save_data]) )