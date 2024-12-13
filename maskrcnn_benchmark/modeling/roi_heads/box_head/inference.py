# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder

# inference过程用到的类
class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    从一系列的类别分类得分，边框回归以及proposals中，计算post-processed boxes,
    以及应用NMS得到最后的结果。
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        post_nms_per_cls_topn=300,
        nms_filter_duplicates=True,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False,
        save_proposals=False,
        custum_eval=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        # 类别得分阈值
        self.score_thresh = score_thresh #0.01
        # nms阈值
        self.nms = nms #0.3
        # 一张图片最后检测结果最大输出box数目
        self.post_nms_per_cls_topn = post_nms_per_cls_topn #300
        self.nms_filter_duplicates = nms_filter_duplicates #T
        self.detections_per_img = detections_per_img #80
        # box编解码器
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg #false，对roi的每个类都回归box，而不是一个roi回归一个box
        self.bbox_aug_enabled = bbox_aug_enabled #false，测试时不会针对box检测进行增强
        self.save_proposals = save_proposals #F
        self.custum_eval = custum_eval #F

    def forward(self, x, boxes, relation_mode=False):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        # 得到box_head结构为每个proposals输出的类别分类结果和box偏移量
        features, class_logits, box_regression = x
        # 进行一个softmax操作
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        # 获取每张图片的size
        image_shapes = [box.size for box in boxes]
        # 获取每一张图片的Proposals数目
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        # 这个地方先不用管它
        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        # add rpn regression offset to the original proposals
        # 给每个Proposal加上偏移量，得到网络微调之后的Proposals
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        ) # tensor of size (num_box, 4*num_cls)
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])
        # 获取分类类别数（包含了背景类别）
        num_classes = class_prob.shape[1]
        # 按照每张图片的Proposals数进行切分
        # 得到的proposals变量维度就是（batch size, 每张图片的Proposals数）
        features = features.split(boxes_per_image, dim=0)
        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        nms_features = []
        # 分别对每一张图片进行操作 因为图片都是按照batch size传入的
        for i, (prob, boxes_per_img, image_shape) in enumerate(zip(
            class_prob, proposals, image_shapes
        )):
            # 将每个加上偏移量（微调）之后的Proposals 按照BoxList的类型进行保存
            # 每个boxlist对象都包含有一张图片中进行ROI_head结构微调过后得到的Proposals
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            assert self.bbox_aug_enabled == False
            if not self.bbox_aug_enabled:
                # If bbox aug is enabled, we will do it later
                # 最后对这些微调之后的Proposals进行筛选
                boxlist, orig_inds, boxes_per_cls = self.filter_results(boxlist, num_classes)
            # add 
            boxlist = self.add_important_fields(i, boxes, orig_inds, boxlist, boxes_per_cls, relation_mode)
            
            results.append(boxlist)
            nms_features.append(features[i][orig_inds])
        
        nms_features = torch.cat(nms_features, dim=0)
        return nms_features, results

    def add_important_fields(self, i, boxes, orig_inds, boxlist, boxes_per_cls, relation_mode=False):
        if relation_mode:
            if not self.custum_eval:
                gt_labels = boxes[i].get_field('labels')[orig_inds]
                # for GQA dataset no need attributes
                if "attributes" in boxes[i].extra_fields:
                    gt_attributes = boxes[i].get_field('attributes')[orig_inds]
                    boxlist.add_field('attributes', gt_attributes)
        
                boxlist.add_field('labels', gt_labels)

            predict_logits = boxes[i].get_field('predict_logits')[orig_inds]
            boxlist.add_field('boxes_per_cls', boxes_per_cls)
            boxlist.add_field('predict_logits', predict_logits)

        return boxlist

    # discarded by kaihua
    def jiaxin_undo_regression(self, i, boxes, orig_inds, boxlist, boxes_per_img):
        # by Jiaxin
        selected_boxes = boxes[i][orig_inds]
        # replace bbox after regression with original bbox before regression
        boxlist.bbox = selected_boxes.bbox
        # add maintain fields
        for field_name in boxes[i].extra_fields.keys():
            if field_name not in boxes[i].triplet_extra_fields:
                boxlist.add_field(field_name, selected_boxes.get_field(field_name))
        # replace background bbox after regression with bbox before regression
        boxes_per_cls = torch.cat((
            boxlist.bbox, boxes_per_img[orig_inds][:,4:]), dim=1).view(len(boxlist), num_classes, 4) # tensor of size (#nms, #cls, 4) mode=xyxy
        boxlist.add_field('boxes_per_cls', boxes_per_cls) # will be used in motif predictor
        return boxlist

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        这个函数就是对一张图片微调之后得到的proposal（box）信息
        类别分类的scores信息 图片的size信息都是整合到一个BoxList对象中去
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("pred_scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        将每一张图片微调之后的Proposals信息、置信度信息、图片尺寸信息都保存在一个BoxList对象当中之后，我们就要通过filter_results()函数来筛选BoxList中的哪些box可以作为结果输出
        注本文后续就将inference过程进行筛选过后得到box的最终结果叫做instances，中间生成的结果还是称作Proposals。
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        # 将BoxList对象中的box(Proposals)取出来  shape is(Proposals数, 类别数*4)
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        boxes_per_cls = boxlist.bbox.reshape(-1, num_classes, 4)
        # 将BoxList对象中的类别置信度得分取出来   shape is(Proposals数, 类别数)
        scores = boxlist.get_field("pred_scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        orig_inds = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        # 判断哪些得分大于阈值
        inds_all = scores > self.score_thresh
        # 通过遍历类别进行筛选  index=0为背景所以跳过  从index=1开始
        for j in range(1, num_classes):
            # 获取得分（类别置信度）大于阈值的索引
            inds = inds_all[:, j].nonzero().squeeze(1)
            # 获取当前类别得分大于阈值的索引
            scores_j = scores[inds, j]
            # 获取上面获取的索引  所对应的box信息
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            # 将该类别（第j类别）的类别得分低于阈值的box信息，图片信息都保存在
            # boxlist_for_class对象中
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            # 给boxlist_for_class添加大于阈值的该类别得分信息
            boxlist_for_class.add_field("pred_scores", scores_j)
            # boxlist_for_class进行NMS操作  操作之后剩余的box都在boxlist_for_class中
            boxlist_for_class, keep = boxlist_nms(
                boxlist_for_class, self.nms, max_proposals=self.post_nms_per_cls_topn, score_field='pred_scores'
            )
            inds = inds[keep]
            # 给剩余下来的box添加第j个类别标签
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "pred_labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            # 进行保存
            result.append(boxlist_for_class)
            orig_inds.append(inds)

        #NOTE: kaihua, according to Neural-MOTIFS (and my experiments, we need remove duplicate bbox)
        if self.nms_filter_duplicates or self.save_proposals:
            assert len(orig_inds) == (num_classes - 1)
            # set all bg to zero
            inds_all[:, 0] = 0 
            for j in range(1, num_classes):
                inds_all[:, j] = 0
                orig_idx = orig_inds[j-1]
                inds_all[orig_idx, j] = 1
            dist_scores = scores * inds_all.float()
            scores_pre, labels_pre = dist_scores.max(1)
            final_inds = scores_pre.nonzero()
            assert final_inds.dim() != 0
            final_inds = final_inds.squeeze(1)

            scores_pre = scores_pre[final_inds]
            labels_pre = labels_pre[final_inds]

            result = BoxList(boxes_per_cls[final_inds, labels_pre], boxlist.size, mode="xyxy")
            result.add_field("pred_scores", scores_pre)
            result.add_field("pred_labels", labels_pre)
            orig_inds = final_inds
        else:
            result = cat_boxlist(result)
            orig_inds = torch.cat(orig_inds, dim=0)
        
        number_of_detections = len(result)
        # Limit to max_per_image detections **over all classes**
        # 如果检测得到的总的intances数目（Proposals）要大于参数设定的最大限制数目
        # 通过置信度排序去除掉一部分
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("pred_scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
            orig_inds = orig_inds[keep]
        return result, orig_inds, boxes_per_cls[orig_inds]
    # 其实关键的思路就是按照每个类取出满足得分阈值要求的Proposals
    # 然后分别对每个类选出的Proposals进行NMS操作
    # 注意：不是对所有类别选出的Proposals，一起做NMS操作


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN #True
    # 和box编解码相关的参数
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS #(10.0, 10.0, 5.0, 5.0)
    box_coder = BoxCoder(weights=bbox_reg_weights)
    # 设置得分阈值 作为哪些box是否输出的依据
    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH #0.01
    # NMS的阈值（用于去除掉一部分box）
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS #0.3
    # 每张图片的检测的最大instance数目
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG #80
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG #False
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED #False
    post_nms_per_cls_topn = cfg.MODEL.ROI_HEADS.POST_NMS_PER_CLS_TOPN #300
    nms_filter_duplicates = cfg.MODEL.ROI_HEADS.NMS_FILTER_DUPLICATES #True
    save_proposals = cfg.TEST.SAVE_PROPOSALS #false
    custum_eval = cfg.TEST.CUSTUM_EVAL #false
    # 生成PostProcessor类对象
    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        post_nms_per_cls_topn,
        nms_filter_duplicates,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled,
        save_proposals,
        custum_eval
    )
    # 返回PostProcessor类对象
    return postprocessor
