# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
     这个类负责将一系列的 bounding boxes 编码, 使之可以被用于训练对应的回归器
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float) 用于对 dw 和 dh 剪枝 值默认为：4.135166556742356
        """
        # 将参数置为成员变量
        self.weights = weights #(10. 10. 5. 5.)
        self.bbox_xform_clip = bbox_xform_clip #4.135166556742356

    def encode(self, reference_boxes, proposals):
        # 根据给定的 reference_boxes 对一系列的 proposals 进行编码
        # proposals(Tensor): 待编码的 boxes
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        TO_REMOVE = 1  # TODO remove
        # 获取宽度(x2-x1)
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        # 获取高度(y2-y1)
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        # 获取中心点坐标
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights
        # 同样的计算逻辑, 获取真实框的宽高和重心坐标
        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        # 计算当前 proposals box 的训练目标, 注意, 不同的 proposals 具有不同的训练目标
        # 最终返回的 tensor 的 shape 为 [n, 4], n 为 proposals 的大小.
        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        # 将中心坐标和宽高的学习目标结合起来
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        # 将编码后的 transformation 形式的 box 转化成 (x1, y1, x2, y2) 形式
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        # 获取真实 box 的宽高和中心坐标
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        # 先对权重解码
        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        # 对 dw, 和 dh 剪枝, 防止向 troch.exp() 传送过大的值
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        # 计算预测的 box 的中心坐标和宽高, 公式说明参见前面的原理讲解
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # 将中心坐标和宽高转化成 (x1, y1, x2, y2) 的表示形式, 然后将其返回
        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes
