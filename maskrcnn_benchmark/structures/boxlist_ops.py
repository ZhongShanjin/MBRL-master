# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import scipy.linalg

from .bounding_box import BoxList

from maskrcnn_benchmark.layers import nms as _box_nms


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode # 缓存当前的模式
    boxlist = boxlist.convert("xyxy") # 转换成指定模式
    boxes = boxlist.bbox # 获取 n*4 的 bbox 列表
    score = boxlist.get_field(score_field) # 获取对应的 socre 列表
    # 调用 _box_nms 执行非极大值抑制抑制
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0: # 只保留 top-k
        keep = keep[: max_proposals]
    # keep为下标列表, 指示了需要保存哪些box, 这里已经重写了 __getitem__ 方法, 因此会根据下标返回 BoxList
    boxlist = boxlist[keep]
    return boxlist.convert(mode), keep


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    # 获取 xywh 形式的bbox列表
    xywh_boxes = boxlist.convert("xywh").bbox
    # torch 的 unbind 函数, 用于 "移除" tensor 的维度,
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]

def squeeze_tensor(tensor):
    tensor = torch.squeeze(tensor)
    try:
        len(tensor)
    except TypeError:
        tensor.unsqueeze_(0)
    return tensor


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def boxlist_union(boxlist1, boxlist2):
    """
    Compute the union region of two set of boxes

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [N,4].

    Returns:
      (tensor) union, sized [N,4].
    """
    # assert len(boxlist1) == len(boxlist2) and (boxlist1.size == boxlist2.size).all()
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    union_box = torch.cat((
        torch.min(boxlist1.bbox[:,:2], boxlist2.bbox[:,:2]),
        torch.max(boxlist1.bbox[:,2:], boxlist2.bbox[:,2:])
        ),dim=1)
    return BoxList(union_box, boxlist1.size, "xyxy")

def boxlist_intersection(boxlist1, boxlist2):
    """
    Compute the intersection region of two set of boxes

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [N,4].

    Returns:
      (tensor) intersection, sized [N,4].
    """
    # assert len(boxlist1) == len(boxlist2) and (boxlist1.size == boxlist2.size).all()
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    inter_box = torch.cat((
        torch.max(boxlist1.bbox[:,:2], boxlist2.bbox[:,:2]),
        torch.min(boxlist1.bbox[:,2:], boxlist2.bbox[:,2:])
        ),dim=1)
    invalid_bbox = torch.max((inter_box[:,0] >= inter_box[:,2]).long(), (inter_box[:,1] >= inter_box[:,3]).long())
    inter_box[invalid_bbox > 0] = 0
    return BoxList(inter_box, boxlist1.size, "xyxy")

# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    # 调用 torch.cat 将数据连接
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    # 确保类型为列表或元组, 且其中元素类型为 BoxList
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    # 确保所有的 BoxList 的 size , mode, 以及 extra_fields 字典的 keys 是相同的
    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields()) # 获取字典的所有 key 值
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    # 调用本文件的 _cat() 方法, 将 bboxes 里面的 BoxList 数据连接成一个 BoxList, 具体解析看下方
    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    # 将各个 BoxList 的 fields 补充上
    for field in fields:
        if field in bboxes[0].triplet_extra_fields:
            triplet_list = [bbox.get_field(field).numpy() for bbox in bboxes]
            data = torch.from_numpy(scipy.linalg.block_diag(*triplet_list))
            cat_boxes.add_field(field, data, is_triplet=True)
        else:
            data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
            cat_boxes.add_field(field, data)

    return cat_boxes

def split_boxlist(bboxes, segs):
    assert isinstance(bboxes, BoxList)
    assert isinstance(segs, (list, tuple))
    size = bboxes.size
    mode = bboxes.mode

    new_boxlists = []
    start_idx = 0
    for each_seg in segs:
        new_boxes = BoxList(bboxes.bbox[start_idx: start_idx + each_seg], size, mode)
        for field in bboxes.fields():
            data = bboxes.get_field(field)[start_idx: start_idx + each_seg]
            new_boxes.add_field(field, data)

        start_idx += each_seg
        new_boxlists.append(new_boxes)

    return new_boxlists