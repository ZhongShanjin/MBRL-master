# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


class Matcher(object):
    # 这个类会给每一个预测 "元素" (如box) 赋值一个 gt "元素".
    # 每一个预测结果最多匹配一个真实框, 每一个真实框可以有多个匹配的预测框.
    # 匹配过程是基于一个 M×N 的匹配矩阵进行的, 矩阵的值代表了行列元素匹配的程度.(如IoU)
    # matcher 对象会返回一个 N 长度的 tensor, N 代表了预测结果的长度,
    # tensor 中的值是一个 0~m-1 的值, 指向了匹配的真实框的下标, 如果没有匹配, 则为负值.
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    这个类是用来给每一个预测的元素（例如box，mask 等等）分配一个GT
    每个预测的元素将有0个或者1个所匹配（0个就是相当于是背景）
    每一个GT将会被对应到0个或者多个预测的元素

    匹配的方式是通过M x N 维度的矩阵，它将predict element(N个元素) 和 GT(M个元素)对应起来
    如果预测的元素为boxes ，这个矩阵将会包含box的IOU值

    matcher的返回值为...
    """
    # 低于阈值
    BELOW_LOW_THRESHOLD = -1
    # 比较模糊的数值
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        参数：
            high_threshold:大于等于这个值的被认为是候选的match
            low_threshold:
            分三种情况
            1) matches >= high_threshold
            2) BETWEEN_THRESHOLDS:matches between [low_threshold, high_threshold)  被赋值为-2
            3) BELOW_LOW_THRESHOLD：matches between [0, low_threshold)   被赋值为-1
        """
        # high_threshold: 置信度大于等于该阈值的 box 被选为候选框. 如 0.7
        # low_threshold: 置信度小于high阈值但是大于等于low阈值的置为 BETWEEN_THRESHOLD,
        # 置信度小于low阈值的置为 BELOW_LOW_THRESHOLD
        # allow_low_quality_matches: 若为真, 则会产生额外一些只有低匹配度的候选框
        # high 阈值必须大于等于 low 阈值
        # 低阈值必须小于高阈值
        assert low_threshold <= high_threshold
        # 给相关参数赋予初始化值
        # 设成员变量
        self.high_threshold = high_threshold #0.5
        self.low_threshold = low_threshold #0.3
        self.allow_low_quality_matches = allow_low_quality_matches #F

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.
            match_quality_matrix是一个MxN维的矩阵，它里面保存的值主要是M个GT元素和N个PT元素
            之间的匹配可信度。（这个矩阵中的值就是GT和PT的IOU值）
            其实就是计算anchors和GT的iou值，之前一直以为是计算Proposals和GT的iou值
        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
         返回值：
             N维的tensor
             N[i]的值为gt的下标，范围为[0, M - 1],或者为一个负值，表示该predict没有匹配的GT
             可以理解为给PT（predict element）都分配了一个对应的GT，如果有PT没有被分配，那么
             该PT位置上的index值用-1表示
        """
        # match_quality_matrix (Tensor[float]): 一个 M×N 的 tensor
        # 包含着 M 个 gt box 和 predicted box 之间的匹配程度
        # 返回值 matches(Tensor[int64]): 一个长度为 N 的 tensor, 其中的元素 N[i]
        # 代表了与第 i 个 predict box 匹配的 gt box 的下标
        # 保证每一张图片里面都至少有一个instance
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            # 在训练过程中, 匹配矩阵中的元素个数不能为 0, 否则, 输出如下错误
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        # 给每一个predict寻找其匹配值最大的值(一个列表，包含每一个proposal与GT的最大值IOU)
        #    以及 其下标（一个列表，每一个proposal所对应的GT下标）
        # 为每个 prediction 找到匹配度最高的 gt candidate
        matched_vals, matches = match_quality_matrix.max(dim=0)
        # 如果允许低于阈值的也是作为候选者，则所有的matches都是
        # 不在乎每个匹配度的实际大小, 保留所有的 prediction 的匹配值
        if self.allow_low_quality_matches:
            all_matches = matches.clone()

        # Assign candidate matches with low quality to negative (unassigned) values
        # 找到哪些index是低于阈值的  哪些index是在阈值之间的
        # 将那些具有低匹配度的值赋值成负数
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        # 给低于阈值的IOU分配 低于阈值对应的值
        # 将 matches 中符合相应条件的值置为对应的值
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD
        # 给处于阈值之间IOU的分配 处于阈值之间的值
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS

        # 如果选项为 True, 则调用类的 set_low_quality_matches_ 函数
        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # 为 predictions 添加仅具有低匹配度的额外的 matches
        # 具体来说, 就是给每一个 gt 找到一个具有最大交并比的 prediction 集合.
        # 对于集合中的每一个 prediction, 如果它还没有与其他 gt 匹配,
        # 则把它匹配到具有最高匹配值的 gt 上.

        # 对于每一个 gt, 找到匹配度最高的 prediction
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        # 找到非零匹配度的下标: (z×2), z 为非零元素的个数.
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]
