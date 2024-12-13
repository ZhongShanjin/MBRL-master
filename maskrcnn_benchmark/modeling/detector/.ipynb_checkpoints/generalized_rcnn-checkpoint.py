# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
        It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes detections / masks from it.
    - build_backbone()主要是创建ResNet+FPN等特征提取网络
    - build_rpn()主要是创建RPN结构
    - build_roi_heads()主要是创建ROI box head  ROI mask head等结构
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        # 创建骨干网络
        self.backbone = build_backbone(cfg)
        # 创建rpn
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        # 创建roi_heads
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model's result (list[BoxList] or dict[Tensor]).
                During training: it returns a dict[Tensor] which contains the losses.(训练阶段返回loss值)
                During testing, it returns list[BoxList] contains additional fieldslike `scores`, `labels` and `mask`
                (for Mask R-CNN models).(测试阶段返回预测的结果（得分， 标签， mask）)

        """
        #训练过程，输入的数据必须有对应的标签，不然没法计算损失
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # 图像经过backbone之后  输出提取的图像特征
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        # 特征经过RPN网络得到proposals和相应的loss值
        # （因此RPN的作用就是获取Proposals和训练时RPN的loss）
        # self.rpn返回值的proposals中是cat_Boxlist得到的结果list(BoxList) shape is （batch size，）
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            # result是检测的结果，detection_losses是检测的损失
            # （如果存在mask分支，result和detection_loss都是包含有mask的检测结果和mask的损失的）
            x, result, detector_losses = self.roi_heads(features, proposals, targets, logger)
        else:
            # RPN-only models don't have roi_heads
            # 如果只提取Proposals（Proposal只是表示可能是需要检测的物体，
            # 具体是什么了类别还不清楚），而不对Proposals进行分类
            x = features
            result = proposals
            detector_losses = {}

        # 将loss值都放到一个字典里面保存
        if self.training:
            losses = {}
            losses.update(detector_losses)
            if not self.cfg.MODEL.RELATION_ON:
                # During the relationship training stage, the rpn_head should be fixed, and no loss. 
                losses.update(proposal_losses)
            return losses

        return result
