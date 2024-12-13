# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

# 该类用于计算 RPN 的损失函数结果
class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    1、anchor 是网络预先固定好位置和大小的bounding box。
    2、proposal是通过网络通过学习给anchor加上一定偏移量得到的bounding box。
    分配GT标签是在anchor的基础上进行判断分配的，而不是在Proposal的基础上进行判断分配的。
    """
    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        # anchor 匹配器，用于匹配anchor和target
        # （因为每一个像素点都包含有9个anchor，所以每一个anchor应该和哪一个target来计算损伤，
        # 这个需要通过proposal_match来进行匹配）
        self.proposal_matcher = proposal_matcher
        # 前景和背景的采集器，因为每一个像素点都对应有9个anchor，
        # 那每一个anchor是当作正样本还是负样本需要进行选择判断
        self.fg_bg_sampler = fg_bg_sampler
        # 边框编码器，用于将anchor进行编码或者解码，用于计算损失
        self.box_coder = box_coder
        # 初始化需要复制的属性
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        # 指定需要放弃的anchor类型
        self.discard_cases = ['not_visibility', 'between_thresholds']

    # 给anchors分配相应的标签
    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        # 计算anchors和GT之间的IOU
        #（维度为MxN  M表示GT的instance数  N表示得到的anchors数）
        match_quality_matrix = boxlist_iou(target, anchor)
        # matched_idxs是一个列表 维度为(N,) 里面的值为GT的instance索引
        # 没有匹配上的值为-1 或-2  低于阈值的值赋予-1  处在阈值之间的值赋予-2
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        # 拷贝一个Boxlist对象  里面的bbox变量拷贝来自target对象
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        # 没有匹配上GT的anchors 都给它们赋予index为0的GT,
        # matched_targets相当于就是一个充当anchors标签的BoxList对象（有box 有label）
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )
            # 得到的matched_idxs中是每一个anchors所匹配好的GT索引
            matched_idxs = matched_targets.get_field("matched_idxs")
            # 得到一个匹配好的anchors的mask，
            # [False, True, True, True, ...]  表示第2，3，4个anchors都已经匹配好GT
            labels_per_image = self.generate_labels_func(matched_targets)
            # 将False 和 True用0 和 1表示
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            # 获取负样本
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            # 计算box的标签
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )
            # 将类别标签和box标签用列表保存
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets


    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        # 获取anchors (BoxList的对象列表)
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        # 给anchors分配标签（GT） 返回的是分配好的类别标签和box标签
        labels, regression_targets = self.prepare_targets(anchors, targets)
        # 按一定比例选取正负样本用于训练阶段计算loss
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness, box_regression = \
                concat_box_prediction_layers(objectness, box_regression)

        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # 计算anchors的偏移量回归loss
        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        # 计算类别的交叉熵loss
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )
        # 返回两个loss
        return objectness_loss, box_loss

# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    # 获得一个bool值得mask  有分配标签的anchor为True 没有对应标签的anchor为False
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_rpn_loss_evaluator(cfg, box_coder):
    """
    RPN模块是通过调用这个函数 用于 计算loss的
    """
    # 匹配器 用于给Proposals分配真实的标签
    # RPNPostProcessor类它只生成了Proposals，但是这些Proposals预测得对不对我们并不知道
    # 所以需要Matcher类来给anchors分配它对应的真实标签，从而知道这些anchors所对应的Proposals预测得对不对
    # 根据配置信息创建 Matcher 实例
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD, # 0.7
        cfg.MODEL.RPN.BG_IOU_THRESHOLD, # 0.3
        allow_low_quality_matches=True,
    )
    # 正负样本筛选器（只有前景和背景两个类别） 选择用于训练的样本
    # 由于anchors的数目比较多，可能会有许多被预测为负样本的情况（毕竟正样本在一张图中是少部分）
    # 为了使得正负样本在训练过程中保持平衡，因此需要该类来进行筛选。
    # 根据配置信息创建一个 BalancedPositiveNegativeSampler 实例
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )
    # 损失的计算器  用于计算RPN阶段的loss值
    # 用于给筛选过后的anchors（加上box的偏移量就得到Proposals）计算其对应的loss。
    # 利用上面创建的实例对象进一步创建 RPNLossComputation 实例
    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels
    )
    return loss_evaluator
