# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# 包的导入
import math

import numpy as np
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    和 nn.ParameterList 差不多, 但是是针对 buffers 的
    """

    def __init__(self, buffers=None):
        # 初始化函数
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        # buffer 扩展
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        # 获取 buffer 长度
        return len(self._buffers)

    def __iter__(self):
        # buffer 迭代器
        return iter(self._buffers.values())


class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set
    of anchors
    对于给定的一系列 image sizes 和 feature maps, 计算对应的 anchors
    """
    # 初始化函数
    def __init__(
        self,
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32),
        straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()

        if len(anchor_strides) == 1:
            # 如果 anchor_strides 的长度为1, 说明没有 fpn 部分, 则直接调用相关函数
            anchor_stride = anchor_strides[0]
            cell_anchors = [
                generate_anchors(anchor_stride, sizes, aspect_ratios).float()
            ]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")

            cell_anchors = [
                generate_anchors(
                    anchor_stride,
                    size if isinstance(size, (tuple, list)) else (size,),
                    aspect_ratios
                ).float()
                for anchor_stride, size in zip(anchor_strides, sizes)
            ]
        # 将 strides, cell_anchors, straddle_thresh 作为 AnchorGenerator 的成员
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh

    # 返回每一个 location 上对应的 anchors 的数量
    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    # 用于生成所有特征图谱的 anchors, 会被 forward 函数调用.
    def grid_anchors(self, grid_sizes):
        # 创建一个空的 anchors 列表
        anchors = []
        # 针对各种组合
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.cell_anchors
        ):
            # 获取 grid 的尺寸和 base_anchors 的 device
            grid_height, grid_width = size
            device = base_anchors.device
            # 按照步长获取偏移量
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            # 获取 y 的偏移量
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            # 创建关于 shifts_y, shifts_x 的 meshgrid
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            # 将二者展开成一维
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        boxlist.add_field("visibility", inds_inside)

    # 定义前向传播过程
    def forward(self, image_list, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                boxlist = BoxList(
                    anchors_per_feature_map, (image_width, image_height), mode="xyxy"
                )
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors

# 根据配置信息创建 AnchorGenerator 对象实例
def make_anchor_generator(config):
    # 定义了 RPN 网络的默认的 anchor 的面积大小
    # 默认值为: (32, 64, 128, 256, 512)
    anchor_sizes = config.MODEL.RPN.ANCHOR_SIZES
    # 定义了 RPN 网络 anchor 的高宽比
    # 默认值为: (0.23232838, 0.63365731, 1.28478321, 3.15089189)
    aspect_ratios = config.MODEL.RPN.ASPECT_RATIOS
    # 定义了 RPN 网络中 feature map 采用的 stride,
    # 对于 FPN 来说, strides 的值应该与 scales 的值匹配
    # 默认值为: (4, 8, 16, 32, 64)
    anchor_stride = config.MODEL.RPN.ANCHOR_STRIDE
    # 移除那些超过图片 STRADDLE_THRESH 个像素大小的 anchors, 起到剪枝作用
    # 默认值为 0, 如果想要关闭剪枝功能, 则将该值置为 -1 或者一个很大的数, 如 100000
    straddle_thresh = config.MODEL.RPN.STRADDLE_THRESH

    if config.MODEL.RPN.USE_FPN:
        # 当使用 fpn 时, 要确保rpn与fpn的相关参数匹配
        assert len(anchor_stride) == len(
            anchor_sizes
        ), "FPN should have len(ANCHOR_STRIDE) == len(ANCHOR_SIZES)"
    else:
        assert len(anchor_stride) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
    # 当获取到相关的参数以后, 创建一个 AnchorGenerator 实例并将其返回
    anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh
    )
    return anchor_generator


def make_anchor_generator_retinanet(config):
    anchor_sizes = config.MODEL.RETINANET.ANCHOR_SIZES
    aspect_ratios = config.MODEL.RETINANET.ASPECT_RATIOS
    anchor_strides = config.MODEL.RETINANET.ANCHOR_STRIDES
    straddle_thresh = config.MODEL.RETINANET.STRADDLE_THRESH
    octave = config.MODEL.RETINANET.OCTAVE
    scales_per_octave = config.MODEL.RETINANET.SCALES_PER_OCTAVE

    assert len(anchor_strides) == len(anchor_sizes), "Only support FPN now"
    new_anchor_sizes = []
    for size in anchor_sizes:
        per_layer_anchor_sizes = []
        for scale_per_octave in range(scales_per_octave):
            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
            per_layer_anchor_sizes.append(octave_scale * size)
        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))

    anchor_generator = AnchorGenerator(
        tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh
    )
    return anchor_generator

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])

# 根据给定的 stride, sizes, aspect_ratio 等参数返回一个 anchor box 组成的矩阵
def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
     1、获取生成 anchors 必要的参数, 包括: stride, sizes, 和 aspect_ratios, 其中, stride 代表特征图谱上的 anchors 的基础尺寸, sizes 代表 anchor 对应在原始图片中的大小(以像素为单位), 因此, 我们容易知道 anchor 在特征图谱上的放缩比例为 sizes/stride, aspect_ratios 代表 anchors 的高宽比, 于是, 最终返回的 anchors 的数量就是 sizes (在特征图谱上固定 base_window 的尺寸, 根据比例的不同来对应不同大小的物体)的数量和 aspect_ratios 数量的乘积;
    2、在获取特征图谱上对应的 base_size(stride)后, 我们将其表示成 [x1, y1, x2, y2](坐标是相对于 anchor 的中心而言的) 的 box 形式. 例如对于 stride=4 的情况, 我们将其表示成 [0, 0, 3, 3], 此部分的实现位于 _generate_anchors(...) 函数中
     3、然后根据 aspect_ratios 的值来获取不同的 anchor boxes 的尺寸, 例如, 对于 stride=4 的 base_anchor 来说, 如果参数 aspect_ratios 为 [0.5, 1.0, 2.0], 那么它就应该返回面积不变, 但是高宽比分别为 [0.5, 1.0, 2.0] 的三个 box 的坐标, 也就是应该返回下面的 box 数组(注意到这里 box 的比例实际上是 [5/2, 1, 2/5], 并不是绝对符合 aspect_ratios, 这是因为像素点只能为整数, 后面还能对这些坐标取整). 这部分的实现位于 _ratio_enum() 函数中;
    [[-1.   0.5  4.   2.5]
     [ 0.   0.   3.   3. ]
    [ 0.5 -1.   2.5  4. ]]
     4、在获取到不同比例的特征图谱上的 box 坐标以后, 我们就该利用 scales = sizes/stride 来将这些 box 坐标映射到原始图像中, 也就是按照对应的比例将这些 box 放大, 对于我们刚刚举的例子 scales = 32/4 = 8 来说, 最终的 box 的坐标如下所示. 这部分的代码实现位于 _scale_num() 函数中.
    [[-22., -10.,  25.,  13.],
     [-14., -14.,  17.,  17.],
    [-10., -22.,  13.,  25.]]
    """
    # 该函数会生成一个 anchor boxes 列表, 列表中的元素为以 (x1, x2, y1, y2) 形式表示的 box
    # 这些 box 的坐标是相对于 anchor 的中心而言的, 其大小为 sizes 数组中元素的平方
    # 这里的默认参数对应的是使用 resnet-C4 作为 backbone 的 faster_rcnn 模型
    # 如果使用了 FPN, 则不同的 size 会对应到不同的特征图谱上, 下面我们利用 FPN 的参数来讲解代码
    # fpn 第一阶段参数值为:(注意sizes必须写成元组或列表的形成)
    # stride=4, sizes=(32,), aspect_ratios=(0.23232838, 0.63365731, 1.28478321, 3.15089189),注意,这里的sizes已经转换成了元组
    return _generate_anchors(
        stride, # stride=4
        np.array(sizes, dtype=np.float) / stride, # sizes / stride = 32 / 4 = [8.]
        np.array(aspect_ratios, dtype=np.float), # [0.23232838, 0.63365731, 1.28478321, 3.15089189]
    )


# 返回 anchor windows ??
def _generate_anchors(base_size, scales, aspect_ratios):
    # 根据调用语句知, 参数值分别为: 4, [8.], [0.23232838, 0.63365731, 1.28478321, 3.15089189]
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    # 首先得到 anchor 的 base box 坐标(相对于 anchor 中心而言), [0, 0, 3, 3]
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    # 根据 base_box 和给定的高宽比, 得到拥有不同高宽比的 anchors,
    # 此处是使 anchor 的比例转化成 [0.23232838 0.63365731 1.28478321 3.15089189], 对应的 box 为:
    # [[-2.   1.   5.   2. ],
    # [-0.5  0.5  3.5  2.5],
    # [ 0.  -0.5  3.   3.5],
    # [ 1.  -1.   2.   4. ]]
    anchors = _ratio_enum(anchor, aspect_ratios)
    # 得到不同高宽比的 anchors 以后, 按照给定的比例(scales)将其缩放到原始图像中,
    # 此处 scales 的值只有一个, 即为 8, 因此, 将上面的 boxes 放大 8 倍(指的是宽高各放大 8 倍, 故总面积会放大64倍), 得到新的 boxes 坐标如下:
    # [[-22., -10.,  25.,  13.],
    # [-14., -14.,  17.,  17.],
    # [-10., -22.,  13.,  25.]]
    # 这里的 vstack 用于将 3 个 1×4 的数组合并成一个 3×4 的数组, 如上所示.
    # anchors[i, :] 代表的是一个 box 的坐标, 如: [-1.  0.5  4.  2.5]
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    # 将numpy数组转换成tensors, 然后返回, anchor的 shape 为: (n, 4), 其中 n 为 anchors 的数量
    return torch.from_numpy(anchors)


# 返回某个 anchor 的宽高以及中心坐标
def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    # 根据左上角和右下角坐标返回该 box 的宽高以及中心点坐标
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

# 给定关于一系列 centers 的宽和高, 返回对应的 anchors
def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    # 将给定的宽, 高以及中心点坐标转化成 (x1, y1, x2, y2) 的坐标形式
    # 这里新增加了一个维度, 以便可以使用 hstack 将结果叠加.
    ws = ws[:, np.newaxis] #eg:[[8.], [5.], [4.], [2.]]
    hs = hs[:, np.newaxis] #eg:[[2.], [3.], [5.], [6.]]
    # 将结果组合起来并返回
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    # 该函数按照给定的 ratios 将 base anchor 转化成具有不同高宽比的多个 anchor boxes, 假设:
    # anchor: [0.  0.  3.  3.]
    # ratios: [0.23232838 0.63365731 1.28478321 3.15089189]
    # 获取 anchor 的宽, 高, 以及中心点的坐标
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    # 获取 anchor 的面积
    size = w * h
    # 根据高宽比获取 size_ratios 变量, 后续会用该变量对 box 的高宽比进行转化
    size_ratios = size / ratios #[68.86803928 25.25024133 12.45346287  5.07792732]
    # ws = sqrt(size) / sqrt(ratios)
    # hs = sqrt(size) * sqrt(ratios)
    # 高宽比 = hs/ws = sqrt(ratios) * sqrt(ratios) = ratios
    # round 代表四舍五入
    ws = np.round(np.sqrt(size_ratios)) #[8. 5. 4. 2.]
    hs = np.round(ws * ratios) #[2. 3. 5. 6.]
    # 根据新的 w 和 h, 生成新的 box 坐标(x1, x2, y1, y2) 并将其返回
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    # anchor: [-1.   0.5  4.   2.5] (举例)
    # scales: 8
    # 获取 anchor 的宽, 高, 以及中心坐标
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    # 将宽和高各放大8倍
    ws = w * scales
    hs = h * scales
    # 根据新的宽, 高, 中心坐标, 将 anchor 转化成 (x1, x2, y1, y2) 的形式
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
