# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

# transpose
# 转换标志
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    BoxList这个类针对每一张图像，都拥有一系列的bounding boxes
    （以一个 Nx4 的tensor形式存在，N是该图片bounding box的数目）
    以及记录图像大小的(width, height)元组。
    此外这个类还有许多的用于bounding box几何变换的方法（例如cropping、scaling、flipping等几何变换）
    该类接受两种不同输入形式的bounding box：
    1、'xyxy', 每一个box都被编码为 'x1' 'y1' 'x2' 'y2'坐标
    2、'xywh', 每一个box都被编码为 'x1' 'y1' 'w' 'h'
    此外，每一个BoxList实例都能够给每一个bounding box 添加任意附加信息，
    例如，标签、可视化、概率得分等等附加信息。
    下面将举一个例子介绍如何创建一个BoxList:
    # 导入所需要的包
    from maskrcnn_benchmark.structures.bounding_box import BoxList, FLIP_LEFT_RIGHT
    import torch

    width = 100
    height = 200
    # boxes的坐标形式 维度：3x4(相当于该图片中有3个bounding box)
    boxes = [
      [0, 10, 50, 50],
      [50, 20, 90, 60],
      [10, 10, 50, 50]
    ]
    # 创建一个含有3个boxes的BoxList对象
    bbox = BoxList(boxes, image_size=(width, height), mode='xyxy')

    # 执行一些变换操作
    bbox_scaled = bbox.resize((width * 2, height * 3))
    bbox_flipped = bbox.transpose(FLIP_LEFT_RIGHT)
    # 给BoxList对象添加标签信息，相当于在字典中添加信息{'labels':labels}
    labels = torch.tensor([0, 10, 1])
    bbox.add_field('labels', labels)
    # BoxList同样支持索引操作
    # 下面就是取索引为0和2的box
    bbox_subset = bbox[[0, 2]]
    """
    def __init__(self, bbox, image_size, mode="xyxy"):
        # 初始化函数
        # bbox (tensor): n×4, 代表n个box, 如: [[0,0,10,10],[0,0,5,5]]
        # image_size: (width, height)
        # 根据 bbox 的数据类型获取对应的 device
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        # 将 bbox 转换成 tensor 类型
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        # bbox 的维度数量必须为2, 并且第二维必须为 4, 即 shape=(n, 4), 代表 n 个 box
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        # 只支持以下两种模式
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        # 为成员变量赋值
        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {} # 以字典结构存储额外信息
        self.triplet_extra_fields = []  # e.g. relation field, which is not the same size as object bboxes and should not respond to __getitem__ slicing v[item]

    # 添加新的键值或覆盖旧的键值
    def add_field(self, field, field_data, is_triplet=False):
        # if field in self.extra_fields:
        #     print('{} is already in extra_fields. Try to replace with new data. '.format(field))
        self.extra_fields[field] = field_data
        if is_triplet:
            self.triplet_extra_fields.append(field)

    # 获取指定键对应的值
    def get_field(self, field):
        return self.extra_fields[field]

    # 判断额外信息中是否存在该键
    def has_field(self, field):
        return field in self.extra_fields

    # 以列表的形式返回所有的键的名称
    def fields(self):
        return list(self.extra_fields.keys())

    # 将另一个 BoxList 类型的额外信息(字典)复制到到的额外信息(extra_fields)中.
    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    # 将当前 bbox 的表示形式转换成参数指定的模式
    def convert(self, mode):
        # 只支持以下两种模式
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        # 调用成员函数, 将坐标表示转化成 (x1,y1,x2,y2) 的形式
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            # 如果模式为 "xyxy", 则直接将 xmin, ymin, xmax, ymax 合并成 n×4 的 bbox
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            # 这里创建了一个新的 BoxList 实例
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            # 否则的话, 就将 xmin, ymin, xmax, ymax 转化成 (x,y,w,h) 后再连接在一起
            TO_REMOVE = 1
            # 创建新的 BoxList 实例
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        # 复制当前实例的 extra_fields 信息到这个新创建的实例当中, 并将这个新实例返回
        bbox._copy_extra_fields(self)
        return bbox

    # 获取 bbox 的 (x1,y1,x2,y2)形式的坐标表示, .split 为 torch 内置函数, 用于将 tensor 分成多块
    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            # x, y 的 shape 为 n × 1, 代表着 n 个 box 的 x, y 坐标
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            # 4 个 tensor 的 shape 均为 n×1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    # 将所有的 boxes 按照给定的 size 和图片的尺寸进行放缩, 创建一个副本存储放缩后的 boxes 并返回
    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """
        # size: 指定放缩后的大小 (width, height)
        # 计算宽和高的放缩比例(new_size/old_size)
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        # 宽高放缩比例相同
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            # 令所有的 bbox 都乘以放缩比例, 不论 bbox 是以 xyxy 形式还是以 xywh 表示
            # 乘以系数就可以正确的将 bbox 的坐标转化到放缩后图片的对应坐标
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            # 复制/转化其他信息
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor) and hasattr(v, "resize"):
                    v = v.resize(size, *args, **kwargs)
                if k in self.triplet_extra_fields:
                    bbox.add_field(k, v, is_triplet=True)
                else:
                    bbox.add_field(k, v)
            return bbox

        # 宽高的放缩比例不同, 因此, 需要拆分后分别放缩然后在连接在一起
        ratio_width, ratio_height = ratios
        # 获取 bbox 的左上角和右下角的坐标
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        # 分别对宽 (xmax, xmin) 和高 (ymax, ymin) 进行放缩
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        # 将左上角和右下角坐标连接起来, 组合放缩后的 bbox 表示
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        # 复制或转化其他信息
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor) and hasattr(v, "resize"):
                v = v.resize(size, *args, **kwargs)
            if k in self.triplet_extra_fields:
                bbox.add_field(k, v, is_triplet=True)
            else:
                bbox.add_field(k, v)

        # 将 bbox 转换成指定的模式(因为前面强制转换成 xyxy 模式了, 所这里要转回去)
        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        # 对 bbox 进行转换(翻转或者旋转90度)
        # methon (int) 此处只能为 0 或 1, 目前仅仅支持两个转换方法

        # 目前仅仅支持 FLIP_LEFT_RIGHT 和 FLIP_TOP_BOTTOM 两种方式
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )
        # 获取图片的宽和高
        image_width, image_height = self.size
        # 获取左上角和右下角的坐标
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        # 将转换后的坐标组合起来形成新的 boxes
        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        # 根据转换后的 boxes 坐标创建一个新的 BoxList 实例, 同时将 extra_fields 信息复制
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            if k in self.triplet_extra_fields:
                bbox.add_field(k, v, is_triplet=True)
            else:
                bbox.add_field(k, v)
        # 将 bbox 的 mode 转换后返回
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        # box 是一个4元组, 指定了希望剪裁的区域的左上角和右下角
        # 获取当前所有 boxes 的最左, 最上, 最下, 最右的坐标
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        # 获取欲剪裁的 box 的宽和高
        w, h = box[2] - box[0], box[3] - box[1]
        # 根据 box 指定的区域, 对所有的 proposals boxes 进行剪裁
        # 即改变其坐标的位置, 如果发现有超出规定尺寸的情况, 则将其截断
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        # 将新的剪裁后的 box 坐标连接起来创建一个新的 BoxList 实例
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        # 复制其他信息
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            if k in self.triplet_extra_fields:
                bbox.add_field(k, v, is_triplet=True)
            else:
                bbox.add_field(k, v)
        # 将 bbox 的模式转换成传入时的模式
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        # device: "cuda:x" or "cpu"
        # 将当前的 bbox 移动到指定的 device 上, 并且重新创建一个新的 BoxList 实例
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        # 深度复制
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            if k in self.triplet_extra_fields:
                bbox.add_field(k, v, is_triplet=True)
            else:
                bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        # item 必须是列表类型
        # 假设 bbox 是一个 BoxList 实例, 那么我们可以利用下面语句得到该实例的子集
        # sub_bbox = bbox[[0,3,4,8]]
        # one_bbox = bbox[[2]]
        # 创建新的子集实例
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        # 复制其他信息
        for k, v in self.extra_fields.items():
            if k in self.triplet_extra_fields:
                bbox.add_field(k, v[item][:,item], is_triplet=True)
            else:
                bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        # 获取当前 BoxList 中含有的 box 的数量
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        #该函数将bbox的坐标限制在image的尺寸内
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            # 该语句会返回一个 n×1 的列表, 对应着 n 个 box, 如果 box 的坐标满足
            # 下面的语句条件, 则对应位为 1, 否则 为 0,
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            # 返回那些对应位为 1 的 box
            return self[keep]
        return self

    def area(self):
        #获取区域面积函数
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            # 一个像素点的面积我们认为是 1, 而不是 0
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            # 直接令 w * h
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy(self):
        return BoxList(self.bbox, self.size, self.mode)

    def copy_with_fields(self, fields, skip_missing=False):
        #连带 BoxList 的 extra_fields 信息进行复制, 返回一个新的 BoxList 实例
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            # 将 fields 包裹在列表里面, 注意 [fields] 和 list(fields) 的区别
            fields = [fields]
        # 遍历 fields 中的所有元素, 并将其添加到当前的 BoxList 实例 bbox 中
        for field in fields:
            if self.has_field(field):
                if field in self.triplet_extra_fields:
                    bbox.add_field(field, self.get_field(field), is_triplet=True)
                else:
                    bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        # 改变 print(BoxList_a) 的打印信息, 使之显示更多的有用信息, 示例如下:
        # BoxList(num_boxes=2, image_width=10, image_height=10, mode=xyxy)
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)
