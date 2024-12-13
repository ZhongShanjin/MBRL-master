# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor


class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression

    RPN中用来进行回归和分类的head模块，在经过了3x3CONV操作之后，就要进行bounding box的回归和2分类任务（有物体还是没有物体）
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        # 使用1*1的卷积将输入的feature的维度转化为预测的anchors的数目（2分类）
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 使用1*1的卷积将输入的feature的维度转化为预测的anchors*4的数目
        #（回归对应到4个坐标点，虽然四个值不是对应四个点，但是可以通过函数转换过去）
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )
        # 初始化 cls__logits和 bbox_pred
        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]
        # 返回值为Proposals（即每一个anchor的二分类结果以及它的坐标偏移量）

        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature

    RPN中用来提取特征的单个卷积层head模块，Backbone提取的图像特征进入RPN模块之后，首先通过一个3x3Conv提取特征
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        # 3*3卷积用于提取特征
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        # 参数初始化
        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
        # 不改变输入输出的特征维度
        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        # 因为batch size的缘故使用这种方式进行计算
        x = [F.relu(self.conv(z)) for z in x]
        # 返回值为经过3x3CONV提取的特征
        return x


@registry.RPN_HEADS.register("SingleConvRPNHead")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
     添加 classification 和 regression heads
    利用分类层和回归层添加一个简单的 RPN heads
     其实就是把RPNHeadConvRegressor类中的相关操作
    整合到一个类当中（先进行3x3CONV 然后进行anchor的bounding box回归和二分类）
     单卷积层的RPN head（里面包含单卷积head 和 分类回归head）
    通过注册器在RPN_HEADS中注册该RPNHead类 方便后面通过字典的形式进行获取
    """

    def __init__(self, cfg, in_channels, mid_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
            cfg: 配置信息
            in_channels (int): 输入特征的通道数
            num_anchors (int): 需要预测的 anchors 的数量
        """
        super(RPNHead, self).__init__()
        # 单层3*3卷积特征提取
        self.conv = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, stride=1, padding=1
        )
        # 2分类
        # objectness 预测层, 输出的 channels 数为 anchors 的数量.(每一点对应 k 个 anchors)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 预测 box 回归的网络层
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        # 对定义的网络层参数进行初始化
        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    # 定义 rpn head 的前向传播过程
    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            # 先执行卷积+激活
            t = F.relu(self.conv(feature))
            # 根据卷积+激活后的结果预测objectness
            logits.append(self.cls_logits(t))
            # 根据卷积+激活后的结果预测 bbox
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    经过RPNHead类得到也是anchors的分类结果和anchors坐标的回归结果
但是并没有涉及应该使用哪些anchors（我们将RPN分类结果为：“有物体” 的anchors，称之为Proposals）用于训练？在训练过程如何进行loss的计算？
    而RPNModule类就是将上述提到问题都进行解决，然后整合的一个模块。
该模块的输入是backbone提取得到的feature
    输出是RPN的proposals和loss值
    """

    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()
        #为每一个像素点生成anchor（每一个像素点一般都会生成9个anchors）
        # 根据配置文件的信息输出对应的 anchor, 详细的实现逻辑需要查看 anchor_generator.py文件
        anchor_generator = make_anchor_generator(cfg)
        # 通过注册器得到cfg中对应的rpn_head
        # cfg.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"
        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        # num_anchors_per_location() 是 AnchorGenerator 的成员函数
        head = rpn_head(
            cfg, in_channels, cfg.MODEL.RPN.RPN_MID_CHANNEL, anchor_generator.num_anchors_per_location()[0]
        )
        # 边框编码器， 主要用于计算边框偏差以及利用偏差计算预测框
        #（就是预测的四个点并不是坐标框的四个点，需要通过函数转化一下）
        # 其主要功能是将 bounding boxes 的表示形式编码成易于训练的形式
        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        # 指定获得预测边框的工具类，将RPN得到的box进行后续处理，用作下一个阶段head的输入
        # 在RPN损失计算部分的anchors和用于后续阶段的Proposals对应的anchors 并不完全一样
        # 挑选用于训练和测试过程的anchors，并返回最后筛选得到的proposals和用于训练的标签。
        # 根据配置信息对候选框进行后处理, 选取合适的框用于训练
        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        # 选取合适的框用于测试
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)
        # 指定RPN误差计算的工具类,用于计算RPN这一部分的loss。
        # 利用得到的box获取损失函数
        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)
        # 设置相应的成员
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)
        输入：
            images:图片的张量列表
                features：backbone所提取的特征图
            targets: 图片的ground truth标签

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        返回值：
            boxes：RPN预测的边框， 一张图对应一个边框列表（边框列表里面有很多边框）
                losses：训练过程所对应的损失（如果是测试阶段这个地方就为空）
        """
        #RPN head得到每一个像素点所对应的多个anchors回归偏量，以及anchors中是否含有物体的二分类结果,
        # objectness是指二分类的结果
        # 利用给定的特征图谱计算相应的 rpn 结果
        objectness, rpn_box_regression = self.head(features)
        # 在图片上生成 anchors
        anchors = self.anchor_generator(images, features)

        # 当处在训练状态时, 调用 _foward_train(), 当处在推演状态时, 调用 _forward_test()
        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    # 训练状态时的前向传播函数
    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            # 当处在 rpn-only 的训练模式时, 网络的 loss 仅仅与rpn的 objectness 和
            # rpn_box_regression values 有关, 因此无需将 anchors 转化成 boxes
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            # 对于 end-to-end 模型来说, anchors 必须被转化成 boxes,
            # 然后采样到目标检测网络的 batch 中用于训练, 注意此时不更新网络参数
            # 需要挑选出一部分box（Proposals）用于下一个阶段的训练
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        # RPN的loss是计算了所有的anchors的loss，而不是仅仅是用于下一阶段boxs（Proposals）的loss
        # 获取损失函数的结果
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )
        # 创建一个loss字典
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
        return boxes, losses

    # 测试状态时的前向传播函数
    def _forward_test(self, anchors, objectness, rpn_box_regression):
        # 将 anchors 转化成对应的 boxes
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            # 对于 end-to-end 模型来说, RPN proposals 仅仅只是网络的一个中间状态,
            # 我们无需将它们以降序顺序存储, 直接返回 FPN 结果即可
            # 但是对于 RPN-only 模式下, RPN 的输出就是最终结果, 我们需要以置信度从高
            # 到低的顺序保存结果并返回.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)

    return RPNModule(cfg, in_channels) #in_channels:256
