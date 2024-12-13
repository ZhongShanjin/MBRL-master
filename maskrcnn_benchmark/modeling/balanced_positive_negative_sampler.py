# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


class BalancedPositiveNegativeSampler(object):
    """
    正负样本的选择器（因为要权衡好正负样本的比例）
     This class samples batches, ensuring that they contain a fixed proportion of positives
    该类用于生成采样 batch, 使得 batch 中的正负样本比例维持一个固定的数
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        batch_size_per_image（在配置文件中设置该参数）
        是指每张图片挑选用于训练的anchors数目（如果实际数目小于这个值，那以实际数目为准）
        postive_fraction 是指batch_size_per_image中正样本个数的比例
        """
        self.batch_size_per_image = batch_size_per_image #256
        self.positive_fraction = positive_fraction #0.5

    def __call__(self, matched_idxs):
        """
        参数:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.
            matched_idxs中包含每一个anchors的label值.  eg:[[1,4,5,0,-1,3,1...],[...],...]
                (0为背景， -1为被忽视的类， positive值为相应的类别号)
            matched idxs: 一个元素类型为 tensor 的列表,
            tensor 包含值 -1, 0, 1. 每个 tensor 都对应这一个具体的图片
            # -1 代表忽略图片中的该样本, 0 代表该样本为负, 正数代表该样本为正

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        每张图片都返回两个二值掩膜列表, 其中一个指示了哪些正样本被采样, 另一个指示了负样本
        """
        # BalancedPositiveNegativeSampler类对象的__call__()函数返回值是：作为正样本的
        # anchors和作为负样本的anchors
        # 例如pos_idx = [[0,1,1,0,1...],...]
        # 表示第一张图片中当作正样本的anchors下标为：1,2,4,...
        # 例如neg_idx = [[1,0,0,0,0,1,...], ....]
        # 表示第一张图片中当作负样本的anchors下标为：0,5,...
        pos_idx = []
        neg_idx = []
        # 批量处理  考虑到batch size维度的缘故
        for matched_idxs_per_image in matched_idxs:
            # 得到正样本的anchors index
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            # 得到负样本的anchors index
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            # 正样本的数目
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            # 从所有的正样本中随机挑选一定数目的正样本 得到的是 postive列表 的index
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            # 从所有的负样本中随机挑选一定数目的负样本
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            # 得到用于训练正样本的 anchors index
            pos_idx_per_image = positive[perm1]
            # 得到用于训练负样本的 anchors index
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            # 将是用来训练的正样本anchors 设置为1
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            # 将是用来训练的负样本anchors 设置为1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        # pos_idx列表中的每一个列表维度可能不太一样
        # （内部每一个列表的维度取决于预测过程中的anchors个数，列表的数目是batch size数目）
        return pos_idx, neg_idx
