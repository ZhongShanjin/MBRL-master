# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn

# 对特征先进行池化，再使用边框分类器进行分类和边框回归器进行回归
# 首先在注册器ROI_BOX_PREDICTOR上注册该类
@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None
        # 输入维度
        num_inputs = in_channels
        # 分类的类别数= 类别数 + 1（背景）
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        # 进行全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 全连接层用于分类
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        # 全连接层用于box的坐标回归
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
        # 类别分类参数初始化
        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        # box回归参数初始化
        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)
    # 执行过程
    def forward(self, x):
        # 平均池化
        x = self.avgpool(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 使用全连接层进行分类
        cls_logit = self.cls_score(x)
        # 使用全连接层进行box回归
        bbox_pred = self.bbox_pred(x)
        # 返回结果
        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES #151
        representation_size = in_channels #4096

        self.cls_score = nn.Linear(representation_size, num_classes) #4096,151
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes #151
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4) #4096,604

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return cls_logit, bbox_pred


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR] #FPNPredictor
    return func(cfg, in_channels) #4096
