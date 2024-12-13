# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .attribute_head.attribute_head import build_roi_attribute_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .relation_head.relation_head import build_roi_relation_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    这个类的作用是将box_head，keypoint_head，mask_head这几个模块都整合在一起
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        # 如果box和mask的head的特征共享，则将box head的features 赋值给mask head
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR: #f
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR: #f
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None, logger=None):
        losses = {}
        # rename x to roi_box_features,
        # if it doesn't increase memory consumption
        # box head的loss
        # self.box就是一个box_head的对象
        # 返回结果是box_head部分提取的特征，detections是检测的结果，loss_box是损失函数
        x, detections, loss_box = self.box(features, proposals, targets)
        if not self.cfg.MODEL.RELATION_ON:
            # During the relationship training stage, the bbox_proposal_network should be fixed, and no loss. 
            losses.update(loss_box)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            # Attribute head don't have a separate feature extractor
            z, detections, loss_attribute = self.attribute(features, detections, targets)
            losses.update(loss_attribute)

        # 如果存在mask 分支
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            # 优化：在训练阶段，如果我们共享了box head 和 mask head的特征提取器，
            # 我们可以重复使用box head所计算的feature用于mask head
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                # 如果mask_feature共享box的特征
                # 就将box_head部分提取的特征赋予mask_features
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            # 训练阶段， self.box() 会返回未经变换的proposals作为检测结果
            # 将mask的的检测结果加入detections中，并计算mask的loss返回。
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)

        if self.cfg.MODEL.RELATION_ON:
            # it may be not safe to share features due to post processing
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_relation = self.relation(features, detections, targets, logger)
            losses.update(loss_relation)

        return x, detections, losses
    # 从上述代码可以看出box_head和mask_head和之前介绍的rpn_heads很相像，
    # 返回的结果都包含有检测的结果和loss
    # rpn_head返回:Proposals(相当于RPN检测的bounding box 和类别结果), rpn_loss
    # box_head返回:提取的特征x, 检测的bounding box和类别分类结果detections, box_loss
    # mask_head返回:提取的特征x, 检测的mask结果并加上之前的box_head的检测结果, mask_loss
    # 因此推断box_head对象和mask_head对象中应该也是包含有相应的loss计算文件和inference文件的

# 创建roi heads
def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON: # RetinaNet 不需要 RoI
        return []

    # 根据配置文件依次添加各个head
    # 从概念上, 下面的 roi 可以同时开启, 互不影响, 但通常只会开启其中一个
    if not cfg.MODEL.RPN_ONLY: # 使用 RPN
        # 添加boxes head
        # 通过build_roi_box_head()创建roi_box_head分支
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON: # 使用 Mask ,F
        # 添加mask head
        # 通过build_roi_mask_head()创建roi_mask_head分支
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON: # 使用 key point ,F
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))
    if cfg.MODEL.RELATION_ON: #T
        roi_heads.append(("relation", build_roi_relation_head(cfg, in_channels)))
    if cfg.MODEL.ATTRIBUTE_ON: #F
        roi_heads.append(("attribute", build_roi_attribute_head(cfg, in_channels)))

    # combine individual heads in a single module
    # 将独立的分支进行合并
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
