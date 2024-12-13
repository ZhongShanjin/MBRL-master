# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
# 导入各种包及函数
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import DFConv2d
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry


# ResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet models
# -----------------------------------------------------------------------------
# ResNet-50 (including all stages)
# ResNet-50 full stages 的2~5阶段的卷积层数分别为:3,4,6,3
ResNet50StagesTo5 = tuple( # 元组内部的元素类型为 StageSpec
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
# ResNet-50-C4, 只使用到第四阶段输出的特征图谱
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# ResNet-101 (including all stages)
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# ResNet-101 up to stage 4 (excludes stage 5)
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# ResNet-50-FPN (including all stages)
# ResNet-50-FPN full stages, 由于 FPN需要用到每一个阶段输出的特征图谱, 故 return_features 参数均为 True
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)
# ResNet-101-FPN (including all stages)
# ResNet-101-FPN full stages 的卷积层数分别为: 3, 4, 23, 3
ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)
# ResNet-152-FPN (including all stages)
ResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        # 初始化
        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        # 将配置文件中的字符串转化成具体的实现, 下面三个分别使用了对应的注册模块, 定义在文件的最后
        # 这里是 stem 的实现, 也就是 resnet 的第一阶段 conv1
        # cfg.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        # resnet conv2_x~conv5_x 的实现
        # eg: cfg.MODEL.CONV_BODY="R-101-FPN"
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        # residual transformation function
        # cfg.MODEL.RESNETS.TRANS_FUNC="BottleneckWithFixedBatchNorm"
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]

        # 获取上面各个组成部分的实现以后, 就可以利用这些实现来构建模型了
        # 构建 stem module(也就是 resnet 的stage1, 或者 conv1)
        # Construct the stem module
        self.stem = stem_module(cfg)

        # Constuct the specified ResNet stages
        # 获取相应的信息来构建 resnet 的其他 stages 的卷积层
        # 当 num_groups=1 时为 ResNet, >1 时 为 ResNeXt
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS #eg:32
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP #eg:8
        # in_channels 指的是向后面的第二阶段输入时特征图谱的通道数,
        # 也就是 stem 的输出通道数, 默认为 64
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS #eg:64
        # 第二阶段输入的特别图谱的通道数
        stage2_bottleneck_channels = num_groups * width_per_group #eg:256
        # 第二阶段的输出, resnet 系列标准模型可从 resnet 第二阶段的输出通道数判断后续的通道数
        # 默认为256, 则后续分别为512, 1024, 2048, 若为64, 则后续分别为128, 256, 512
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS #eg:256
        # 创建一个空的 stages 列表和对应的特征图谱字典
        self.stages = []
        self.return_features = {}
        #eg:(StageSpec(index=1, block_count=3, return_features=True),
        # StageSpec(index=2, block_count=4, return_features=True),
        # StageSpec(index=3, block_count=23, return_features=True),
        # StageSpec(index=4, block_count=3, return_features=True))
        #构建4个layer
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            # 计算每个stage的输出通道数, 每经过一个stage, 通道数都会加倍
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            # 计算输入图谱的通道数
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            # 计算输出图谱的通道数
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index -1]
            # 当获取到所有需要的参数以后, 调用本文件的 `_make_stage` 函数,
            # 该函数可以根据传入的参数创建对应 stage 的模块(注意是module而不是model)
            module = _make_stage(
                transformation_module, #BottleneckWithFixedBatchNorm
                in_channels, # 输入的通道数
                bottleneck_channels, # 中间的通道数
                out_channels, # 输出的通道数
                stage_spec.block_count, #当前stage的卷积层数量
                num_groups, # ResNet时为1, ResNeXt时>1
                cfg.MODEL.RESNETS.STRIDE_IN_1X1, #eg:False
                # 当处于 stage3~5时, 需要在开始的时候使用 stride=2 来下采样
                first_stride=int(stage_spec.index > 1) + 1, #eg:(1,2,2,2)
                dcn_config={
                    "stage_with_dcn": stage_with_dcn, #False
                    "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN, #False
                    "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS, #1
                }
            )
            # 下一个 stage 的输入通道数即为当前 stage 的输出通道数
            in_channels = out_channels
            # 将当前stage模块添加到模型中
            self.add_module(name, module)
            # 将stage的名称添加到列表中
            self.stages.append(name)
            # 将stage的布尔值添加到字典中
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        # 根据配置文件的参数选择性的冻结某些层(requires_grad=False)
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        # 将指定的参数置为: requires_grad = False
        # 根据给定的参数冻结某些层的参数更新
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            # 将 m 中的所有参数置为不更新状态.
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        # 定义 resnet 的前向传播过程
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


class ResNetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
        res2_out_channels=256,
        dilation=1,
        dcn_config={}
    ):
        super(ResNetHead, self).__init__()

        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x

# 创建 ResNet 的 residual-block
def _make_stage(
    transformation_module, #BottleneckWithFixedBatchNorm
    in_channels, #64
    bottleneck_channels, #256
    out_channels, #256
    block_count, #3
    num_groups, #32
    stride_in_1x1, #False
    first_stride, #1
    dilation=1, #1
    dcn_config={} #{'stage_with_dcn': False, 'with_modulated_dcn': False, 'deformable_groups': 1}
):
    blocks = [] #BottleneckWithFixedBatchNorm(downsample->conv1->bn1->conv2->bn2->conv3->bn3)
    stride = first_stride
    for _ in range(block_count):
        blocks.append(
            transformation_module( #BottleneckWithFixedBatchNorm
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
        )
        stride = 1
        in_channels = out_channels
    return nn.Sequential(*blocks)


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func, #FrozenBatchNorm2d
        dcn_config
    ):
        super(Bottleneck, self).__init__()

        # downsample: 当 bottleneck 的输入和输出的 channels 不相等时, 则需要采用一定的策略,也就是利用参数矩阵映射使得输入输出的 channels 相等
        self.downsample = None
        # 当输入输出通道数不同时, 额外添加一个 1×1 的卷积层使得输入通道数映射成输出通道数
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),  # 后街一个固定参数的 BN 层 FrozenBatchNorm2d
            )
            for modules in [self.downsample,]: #就是上面定义的一个卷积层和BN层
                for l in modules.modules(): #就是Sequential的内容，0是Conv2d,1是BN层
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1 # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have stride in the 3x3 conv
        # 在 resnet 原文中, 会在 conv3_1, conv4_1, conv5_1 处使用 stride=2 的卷积
        # 而在 fb.torch.resnet 和 caffe2 的实现中, 是将之后的 3×3 的卷积层的 stride 置为2
        # 下面中的 stride 虽然默认值为1, 但是在函数调用时, 如果stage为3~5, 则会显示置为2
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        # 当获取到当前stage所需的参数后, 就创建相应的卷积层, 创建原则参见 resnet50 的定义
        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels) # 后接一个固定参数的 BN 层
        # TODO: specify init for the above
        with_dcn = dcn_config.get("stage_with_dcn", False) #False
        if with_dcn:
            deformable_groups = dcn_config.get("deformable_groups", 1)
            with_modulated_dcn = dcn_config.get("with_modulated_dcn", False)
            self.conv2 = DFConv2d(
                bottleneck_channels,
                bottleneck_channels,
                with_modulated_dcn=with_modulated_dcn,
                kernel_size=3,
                stride=stride_3x3,
                groups=num_groups,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False
            )
        else:
            # 创建 bottleneck 的第二层卷积层
            self.conv2 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                padding=dilation,
                bias=False,
                groups=num_groups,
                dilation=dilation
            )
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)
        self.bn2 = norm_func(bottleneck_channels) # 后接一个 BN 层

        # 创建 bottleneck 的最后一个卷积层, padding默认为1
        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)


        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        # 执行一次forward, 相当于执行一次 bottleneck,
        # 默认情况下, 具有三个卷积层, 一个恒等连接, 每个卷积层之后都带有 BN 和 relu 激活
        # 注意, 最后一个激活函数要放在恒等连接之后
        identity = x # 恒等连接, 直接令残差等于x即可

        # conv1, bn1
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        # conv2, bn2
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        # conv3, bn3
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            # 如果输入输出的通道数不同, 则需要通过映射使之相同.
            identity = self.downsample(x)

        out += identity # H = F + x
        out = F.relu_(out) # 最后进行激活

        return out  # 返回带有残差项的卷积结果


class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        # out_channels=64
        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        # 输入的 channels 为 3, 输出为 64
        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        # 使用固定参数的 BN 层
        self.bn1 = norm_func(out_channels)
        # l = Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 初始化Conv2d层的weight参数
        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x) # 原地激活, 因为不含参数, 因此不放在模型定义中, 而放在 forward 中实现
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class BottleneckWithFixedBatchNorm(Bottleneck):
    # 使用固定的BN
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={}
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=FrozenBatchNorm2d,
            dcn_config=dcn_config
        )


class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )


class BottleneckWithGN(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={}
    ):
        super(BottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm,
            dcn_config=dcn_config
        )


class StemWithGN(BaseStem):
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)


_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})

_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C4": ResNet101StagesTo4,
    "R-101-C5": ResNet101StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
    "R-50-FPN-RETINANET": ResNet50FPNStagesTo5,
    "R-101-FPN": ResNet101FPNStagesTo5,
    "R-101-FPN-RETINANET": ResNet101FPNStagesTo5,
    "R-152-FPN": ResNet152FPNStagesTo5,
})
