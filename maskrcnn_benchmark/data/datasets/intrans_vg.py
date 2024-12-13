import os
import sys
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
from copy import deepcopy

from itertools import product
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from .visual_genome import load_info, load_image_filenames, correct_img_info, get_VG_statistics
import pickle
from collections import Counter
from maskrcnn_benchmark.config.defaults import _C as config

BOX_SCALE = 1024  # Scale at which we have the boxes

def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    # print('boxes1: ', boxes1.shape)
    # print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:, :, :2], boxes2.reshape([1, num_box2, -1])[:, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:, :, 2:], boxes2.reshape([1, num_box2, -1])[:, :, 2:])  # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter

class InTransDataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                 filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, custom_eval=False,
                 custom_path='', custom_bbox_path='', distant_supervsion_file=None, specified_data_file=None):
        """
            The dataset to conduct internal transfer
            or used for training a new model based on tranferred dataset
            Parameters:
                split: Must be train, test, or val
                img_dir: folder containing all vg images
                roidb_file:  HDF5 containing the GT boxes, classes, and relationships
                dict_file: JSON Contains mapping of classes/relationships to words
                image_file: HDF5 containing image filenames
                filter_empty_rels: True if we filter out images without relationships between
                                 boxes. One might want to set this to false if training a detector.
                filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
                num_im: Number of images in the entire dataset. -1 for all images.
                num_val_im: Number of images in the validation set (must be less than num_im
                   unless num_im is -1.)
                specified_data_file: pickle file constains training data
        """
        assert split in {'train'}
        assert flip_aug is False
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        # apply predicate reweight or not
        self.rwt = config.IETRANS.RWT

        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(
            dict_file)  # contiguous 151, 51 containing __background__
        self.num_rel_classes = len(self.ind_to_predicates)
        self.categories = {i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        self.custom_eval = custom_eval
        self.data = pickle.load(open(specified_data_file, "rb"))
        print(specified_data_file)
        self.img_info = [{"width":x["width"], "height": x["height"]} for x in self.data]
        self.filenames = [x["img_path"] for x in self.data]

        if self.rwt:
            # construct a reweighting dic
            self.reweighting_dic = self._get_reweighting_dic()

    def __getitem__(self, index):
        img = Image.open(self.data[index]["img_path"]).convert("RGB")
        target = self.get_groundtruth(index)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # append current raw data
        # it is unusable under most conditions
        target.add_field("cur_data", self.data[index])
        return img, target, index

    def _get_reweighting_dic(self):
        """
        weights for each predicate
        weight is the inverse frequency normalized by the median
        Returns:
            {1: f1, 2: f2, ... 50: f50}
        """
        rels = [x["relations"][:, 2] for x in self.data]
        rels = [int(y) for x in rels for y in x]
        rels = Counter(rels)
        rels = dict(rels)
        rels = [rels[i] for i in sorted(rels.keys())]
        vals = sorted(rels)
        rels = torch.tensor([-1.]+rels)
        rels = (1./rels) * np.median(vals)
        return rels

    def get_statistics(self, no_matrix=False):
        if no_matrix:
            return {
                'fg_matrix': None,
                'pred_dist': None,
                'obj_classes': self.ind_to_classes,
                'rel_classes': self.ind_to_predicates,
                'att_classes': self.ind_to_attributes,
            }

        fg_matrix, bg_matrix = get_VG_statistics(img_dir=self.img_dir, roidb_file=self.roidb_file,
                                                 dict_file=self.dict_file,
                                                 image_file=self.image_file, must_overlap=True)

        # num_obj_classes = len(self.ind_to_classes)
        # num_rel_classes = len(self.ind_to_predicates)
        # fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
        # bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)
        #
        # for ex_ind in tqdm(range(len(self.data))):
        #     gt_classes = self.data[ex_ind]['labels'].copy()
        #     gt_relations = self.data[ex_ind]['relations'].copy()
        #     gt_boxes = self.data[ex_ind]['boxes'].copy()
        #
        #     # For the foreground, we'll just look at everything
        #     o1o2 = gt_classes[gt_relations[:, :2]]
        #     for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
        #         fg_matrix[o1, o2, gtr] += 1
        #     # For the background, get all of the things that overlap.
        #     o1o2_total = gt_classes[np.array(
        #         box_filter(gt_boxes, must_overlap=True), dtype=int)]
        #     for (o1, o2) in o1o2_total:
        #         bg_matrix[o1, o2] += 1

        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_attributes,
        }
        return result

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        for file_name in os.listdir(path):
            self.custom_files.append(os.path.join(path, file_name))
            img = Image.open(os.path.join(path, file_name)).convert("RGB")
            self.img_info.append({'width': int(img.width), 'height': int(img.height)})

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, flip_img=False):
        cur_data = self.data[index]
        w, h = cur_data['width'], cur_data['height']
        relation_tuple = cur_data["relations"]
        pairs = relation_tuple[:, :2]
        rel_lbs = relation_tuple[:, 2]
        relation_labels = torch.zeros((rel_lbs.shape[0], self.num_rel_classes))
        # relation_labels: [0, 0, 0, 1, ..., 0]
        relation_labels[torch.arange(0, relation_labels.size(0)), rel_lbs] = 1.
        #ori_relation
        if 'ori_relations' not in cur_data:
            cur_data['ori_relations'] = deepcopy( cur_data['relations'] )
        if 'grain_size' not in cur_data:
            cur_data['grain_size'] = np.ones( len(cur_data['relations']) )
        ori_relation_tuple = deepcopy( cur_data['ori_relations'] )
        ori_rel_lbs = ori_relation_tuple[:, 2]
        ori_relation_labels = torch.zeros((ori_rel_lbs.shape[0], self.num_rel_classes))
        ori_relation_labels[torch.arange(0, ori_relation_labels.size(0)), ori_rel_lbs] = 1.
        # grain_size
        grain_size = deepcopy(cur_data['grain_size'])
        grain_size = torch.unsqueeze(torch.from_numpy(grain_size).float(), 1)
        # reweighting
        if self.rwt:
            assert ~(rel_lbs == 0).any(), rel_lbs
            weights = self.reweighting_dic[rel_lbs]
            ori_weights = self.reweighting_dic[ori_rel_lbs]
            # put the weight at the predicate 0, which is background
            # the loss function will extract this
            # relation_labels: [weight, 0, 0, 1, ..., 0]
            relation_labels[:, 0] = -weights
            ori_relation_labels[:, 0] = -ori_weights

        box = torch.from_numpy(cur_data['boxes']).reshape(-1, 4)  # guard against no boxes
        target = BoxList(box, (w, h), 'xyxy')  # xyxy
        # object labels
        target.add_field("labels", torch.from_numpy(cur_data['labels']))
        # object attributes
        target.add_field("attributes", torch.zeros((box.size(0), 10)))
        # relation pair indexes
        target.add_field("relation_pair_idxs", torch.from_numpy(pairs).long())
        target.add_field("relation_labels", relation_labels)
        target.add_field("ori_relation_labels", ori_relation_labels)
        target.add_field("grain_size", grain_size)
        target.add_field("train_data", cur_data)
        return target

    def __len__(self):
        if self.custom_eval:
            return len(self.custom_files)
        return len(self.data)

