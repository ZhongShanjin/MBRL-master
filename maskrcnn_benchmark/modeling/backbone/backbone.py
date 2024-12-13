# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict # 导入有序字典

from torch import nn

# 注册器, 用于管理 module 的注册, 使得可以像使用字典一样使用 module
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import vgg

@registry.BACKBONES.register("VGG-16")
def build_vgg_fpn_backbone(cfg):
    body = vgg.VGG16(cfg)
    out_channels = cfg.MODEL.VGG.VGG16_OUT_CHANNELS
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = out_channels
    return model

# 相当于在BACKBONES的字典中注册了"R-50-C4"、"R-50-C5"、"R-101-C4"、"R-101-C5"这几个key
# 它们对应的value都是 build_resnet_backbone()这个函数
# 创建 resnet 骨架网络, 根据配置信息会被后面的 build_backbone() 函数调用
@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    # 通过cfg来决定resnet是50还是101
    body = resnet.ResNet(cfg)  # resnet.py 文件中的 class ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)])) # 利用 nn.Sequential 定义模型
    # 通过cfg来决定输出是C4还是C5
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model

# 创建 fpn 网络, 根据配置信息会被下面 build_backbone 函数调用
@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):  # 先创建 resnet 网络
    body = resnet.ResNet(cfg)
    # 获取 fpn 所需的channels参数
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN( # 利用 fpn.py 文件夹的 class FPN 创建 fpn 网络
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    """
    backbone的类型通过配置文件中的CONV_BODY来决定
     registry.BACKBONES[***]是获取对应的注册器字典中的value (当前场景中value是一个函数)
    然后对获取的函数输入形参cfg  返回一个backbone模型对象。
     等价于如下
    foo = registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY]
     backbone = foo(cfg)
    return backbone
    """
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
