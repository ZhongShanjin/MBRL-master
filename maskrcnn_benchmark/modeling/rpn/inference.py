# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# 导入各种包及函数
import torch

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

from ..utils import cat
from .utils import permute_and_flatten

class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
     1、在所有anchors中筛选出top_k个anchors，top_k由参数pre_nms_top_n决定。
    2、将筛选之后得到的anchors和它对应的box回归值进行结合，得到对应的Proposals。(box回归值就是RPN预测的anchors的偏移量)
     3、将面积小于min_size的Proposals去除掉，min_size由参数min_size决定。
    4、通过NMS操作，对Proposals进行筛选，得到top_n个Proposals，top_n由参数post_nms_top_n决定。
     5、最后将top_n个Proposals和target标签一并作为结果返回（此时标签和Proposals的还未一一对应上）。
    """
    # 初始化函数中主要是定义类变量，下面对几个类变量的意义做一定解释
    # 在将 proposals 喂到网络的 heads 之前, 先对 RPN 输出的 boxes 执行后处理
    # 该类主要完成对 RPN boxes 的后处理功能(在将 boxes 送到 heads 之前执行)
    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
        fpn_post_nms_per_batch=True,
        add_gt=True,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        # 将传送进来的参数都变成成员变量
        # 通过二分类置信度挑选的top_k个 anchors数目
        self.pre_nms_top_n = pre_nms_top_n
        # 通过NMS挑选top_n个Proposals的数目
        self.post_nms_top_n = post_nms_top_n
        # NMS方法的阈值
        self.nms_thresh = nms_thresh
        # 去除Proposals面积小于min_size的proposals
        self.min_size = min_size
        self.add_gt = add_gt

        # box_coder可以将RPN得到的regression偏移量添加到anchors上去
        # 创建一个 BoxCoder 实例
        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.fpn_post_nms_per_batch = fpn_post_nms_per_batch

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        这个函数返回结果并没有将Proposals和target标签一一对应上
        即并不知道哪个Proposal对应哪个target标签，只是把它们放在一起保存而已。
        """
        # Get the device we're operating on
        # 将真实的边框标签 targets 添加到当前的 BoxList 列表数据中.
        # 获取当前正在操作的设备
        device = proposals[0].bbox.device
        # 拷贝一个dataset中获得的boxlist对象（dataset中的target）（fields不进行拷贝）
        # 调用 BoxList 的 copy_with_fields 方法进行深度复制, gt_boxes 是一个列表
        # 其元素的类型是 BoxList
        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        # gt_boxes中没有任何的field
        # 添加一个objectness的fields
        # BoxList是项目内设的一个类，add_field（）方法就是添加字典数据的过程
        # 下面这个就是在gt_box中内置字典中，添加一个key为objectness，value为[1...1]的数据
        # 添加一个字典键, "objectness", 值为当前 BoxList 元素中的 box 的数量长度的一维 tensor
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))
        # 调用 boxlist_ops.py 中的 cat_boxlist 函数将 proposal 和 gt_box 合并成一个 BoxList
        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        N: batch size
         A: num of anchor
        H: height of image
         W: width of image
        关于RPN得到的anchors数目，我还是简单的讲一下，
         RPN是流程一般是给每一个像素点都生成9个anchors（num of anchor），
        因此每一张图片中，总的anchors数目为:A * H * W
         H, W 代表特征图谱的高和宽
        """
        # 获取当前的设备
        device = objectness.device
        # 获取 objectness 的 shape
        N, A, H, W = objectness.shape

        # put in the same format as anchors
        # objectness的维度为 (N, A*H*W)
        # 将格式转换成和 anchors 相同的格式, 先改变维度的排列, 然后改变 shape 的形式
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        # sigmoid 归一化
        objectness = objectness.sigmoid()
        # 相似的操作, 应用在 box_regression 上
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        # 计算 anchors 的总数量
        num_anchors = A * H * W
        # 确保 pre_nms_top_n 不会超过 anchors 的总数量, 以免产生错误
        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        # 通过对objectness置信度排序，挑选出top_n个anchors（objectness 和 box_regression）
        # 调用 PyTorch 的 topk 函数, 该函数返回两个列表, 一个是 topk 的值, 一个是对应下标
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)
        # 创建 batch 的下标, shape 为 N×1, 按顺序递增, 如:[[0],[1],...,[N-1]]
        batch_idx = torch.arange(N, device=device)[:, None]
        # 获取所有 batch 的 top_k box
        box_regression = box_regression[batch_idx, topk_idx]
        # 获取所有 anchor 的尺寸
        image_shapes = [box.size for box in anchors]
        # 获取所有 box, 将 anchors 连接成一个列表
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        # 重新按照 batch 划分, 同时获取每个 batch 的 topk
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]
        # 给筛选之后得到的anchors添加regression偏移量 得到Proposals
        # 将最终的结果解码成方便表示的形式(原本为方便训练的形式)
        proposals = self.box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4)
        )

        proposals = proposals.view(N, -1, 4)

        result = [] # 组建结果并返回
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            # 根据当前的结果创建一个 BoxList 实例
            boxlist = BoxList(proposal, im_shape, mode="xyxy")
            # 添加 score
            boxlist.add_field("objectness", score)
            # 防止 box 超出 image 的边界
            boxlist = boxlist.clip_to_image(remove_empty=False)
            # 去除面积小于min_size的Proposals
            boxlist = remove_small_boxes(boxlist, self.min_size)
            # 对Proposals进行NMS操作得到最后的Proposals
            boxlist, _ = boxlist_nms(
                boxlist,
                self.nms_thresh,
                max_proposals=self.post_nms_top_n,
                score_field="objectness",
            )
            result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        # 创建一个空的 box 列表
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        # 对anchors进行采样 得到Proposals（在所有anchors中筛选出top_k个anchors，top_k由参数pre_nms_top_n决定。
        # 将筛选之后得到的anchors和它对应的box回归值进行结合，得到对应的Proposals。(box回归值就是RPN预测的anchors的偏移量)
        # 将面积小于min_size的Proposals去除掉，min_size由参数min_size决定。
        # 通过NMS操作，对Proposals进行筛选，得到top_n个Proposals，top_n由参数post_nms_top_n决定。）
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        # 应该是将FPN每一层的Proposals都添加在一起（具体还没弄明白）
        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        # 将得到的Proposals和真实的标签放到一起保存
        if self.training and (targets is not None) and self.add_gt:
            boxlists = self.add_gt_proposals(boxlists, targets)
        # 返回Proposals
        return boxlists

    def select_over_all_levels(self, boxlists):
        # 在训练阶段和测试阶段的行为不同, 在训练阶段, post_nms_top_n 是在所有的 proposals 上进行的,
        # 而在测试阶段, 是在每一个图片上的 proposals 上进行的
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # NOTE: it should be per image, and not per batch. However, to be consistent 
        # with Detectron, the default is per batch (see Issue #672)
        if self.training and self.fpn_post_nms_per_batch:
            # 连接 "objectness"
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            # 获取box的数量
            box_sizes = [len(boxlist) for boxlist in boxlists]
            # 防止 post_nms_top_n 超过 anchors 总数, 产生错误
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            # 获取 topk 的下标
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            # 获取所有满足条件的box
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    # 设置nms之后保留的Proposals数目（有FPN的情况）
    # rpn_box_coder: BoxCoder 实例
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN #1000
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST #1000

    # 设置 通过RPN输出anchor的二分类置信度进行筛选 最后保留的anchor数目（有点拗口~）
    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN #6000
    # 设置nms之后保留的Proposals数目
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN #1000
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST #6000
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST #1000
    fpn_post_nms_per_batch = config.MODEL.RPN.FPN_POST_NMS_PER_BATCH #False
    # 设置NMS阈值
    nms_thresh = config.MODEL.RPN.NMS_THRESH #0.7
    # 设置最小的Proposals面积大小
    min_size = config.MODEL.RPN.MIN_SIZE #0
    add_gt = config.MODEL.ROI_RELATION_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN #True
    # 根据配置参数创建一个 RPNPostProcessor 实例
    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        fpn_post_nms_per_batch=fpn_post_nms_per_batch,
        add_gt=add_gt,
    )
    # 返回RPNPostProcessor类对象
    return box_selector
