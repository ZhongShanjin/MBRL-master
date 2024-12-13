# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNSampling(object):
    """
    Sampling RoIs
    """
    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # 定义用于Proposals标签匹配的 匹配器
        self.proposal_matcher = proposal_matcher
        # 定义用于正负样本筛选的 筛选器
        self.fg_bg_sampler = fg_bg_sampler
        # 定义box的编解码器
        self.box_coder = box_coder

    def match_targets_to_proposals(self, proposal, target):
        # gt 和 RPN输出的Proposals之间的 IOU矩阵
        match_quality_matrix = boxlist_iou(target, proposal)
        # 预测边框和对应的gt的索引， 背景边框为-2 ， 模糊边框为-1
        # eg:matched_idxs[4] = 6 :表示第5个预测边框所分配的GT的id为6
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        # 获得 GT 的类别标签
        target = target.copy_with_fields(["labels", "attributes"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        # 将所有的背景边框和模糊边框的标签都对应成第一个gt的标签
        # 其实就是将target中的box 和label按照Proposals的对应顺序重新排序的一个过程，
        # 将target中box顺序和matched_idxs中的GT的id顺序保持一致
        matched_targets = target[matched_idxs.clamp(min=0)]
        # 将对应的列表索引添加至gt列表中
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    # 准备类别标签和box偏移量标签
    def prepare_targets(self, proposals, targets):
        # 类别标签列表
        labels = []
        attributes = []
        # 回归box标签列表
        regression_targets = []
        matched_idxs = []
        # 分别对每一张图片进行操作
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs_per_image = matched_targets.get_field("matched_idxs")
            
            # 获取每一个target所对应的label标签
            labels_per_image = matched_targets.get_field("labels")
            attris_per_image = matched_targets.get_field("attributes")
            labels_per_image = labels_per_image.to(dtype=torch.int64)
            attris_per_image = attris_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            # 背景标签
            bg_inds = matched_idxs_per_image == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0
            attris_per_image[bg_inds,:] = 0

            # Label ignore proposals (between low and high thresholds)
            # 被忽视的样本
            ignore_inds = matched_idxs_per_image == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            # 计算偏移量target  因为网络预测的结果是偏移量，所以需要生成偏移量标签
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            # 对生成好的类别标签和偏移量标签进行保存
            labels.append(labels_per_image)
            attributes.append(attris_per_image)
            regression_targets.append(regression_targets_per_image)
            matched_idxs.append(matched_idxs_per_image)

        #返回为Proposals匹配好的类别标签和box偏移量标签
        return labels, attributes, regression_targets, matched_idxs

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
         Note: this function keeps a state.
        正负样本的筛选

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        # 获取Proposals分配好的标签
        labels, attributes, regression_targets, matched_idxs = self.prepare_targets(proposals, targets)
        # 获取被分配为正负样本的索引  由BalancedPositiveNegativeSampler类进行分配
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, attributes_per_image, regression_targets_per_image, matched_idxs_per_image, proposals_per_image in zip(
            labels, attributes, regression_targets, matched_idxs, proposals
        ):
            # 给BoxList类型的Proposals添加标签信息
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field("attributes", attributes_per_image)
            proposals_per_image.add_field("regression_targets", regression_targets_per_image)
            proposals_per_image.add_field("matched_idxs", matched_idxs_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        # 对BoxList类型的Proposals进行正负样本筛选（对应的标签也会一并被筛选出来）
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image
        # 得到筛选之后的Proposals（BoxList对象 其中包含有label信息）
        return proposals

    def assign_label_to_proposals(self, proposals, targets):
        for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
            match_quality_matrix = boxlist_iou(target, proposal)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # Fast RCNN only need "labels" field for selecting the targets
            # for GQA dataset no need attributes
            if "attributes" in target.extra_fields:
                target = target.copy_with_fields(["labels", "attributes"])
            else:
                target = target.copy_with_fields(["labels"])
            matched_targets = target[matched_idxs.clamp(min=0)]
            
            labels_per_image = matched_targets.get_field("labels").to(dtype=torch.int64)
            if "attributes" in target.extra_fields:
                attris_per_image = matched_targets.get_field("attributes").to(dtype=torch.int64)
                attris_per_image[matched_idxs < 0, :] = 0
                proposals[img_idx].add_field("attributes", attris_per_image)

            labels_per_image[matched_idxs < 0] = 0
            proposals[img_idx].add_field("labels", labels_per_image)

        return proposals


def make_roi_box_samp_processor(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, #0.5
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD, #0.3
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS #(10 10 5 5)
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler( #256, 0.5
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    samp_processor = FastRCNNSampling(
        matcher,
        fg_bg_sampler,
        box_coder,
    )

    return samp_processor
