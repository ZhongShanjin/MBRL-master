# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# 导入各种包及函数
import logging
import time
import os
import pickle
import json
import torch
from tqdm import tqdm
import numpy as np
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

def modify_logits(tg_data, pred_data):
    # just append the logits
    cur = {'image_path':tg_data.extra_fields['img_path']}
    cur["bbox"] = tg_data.bbox.cpu().numpy()
    cur["size"] = tg_data.size
    cur["labels"] = tg_data.extra_fields['labels'].cpu().numpy()
    cur["relation_tuple"] = tg_data.extra_fields['relation_tuple'].cpu().numpy()
    rpi = pred_data.extra_fields['rel_pair_idxs'].cpu().numpy()
    crt = cur["relation_tuple"][:,:2]
    pairs = []
    for crti in crt:
        pairs.extend(np.where(np.all(rpi == crti, axis=1))[0])
    cur["relation_logits"] = pred_data.extra_fields['pred_rel_scores'].cpu().numpy()[pairs]
    cur['pred_rel'] = np.argmax(cur["relation_logits"][:,1:], axis=1) + 1
    cur['gt_rel'] = cur["relation_tuple"][:,2]
    cur['gt_pair'] = cur["relation_tuple"][:,:2]
    return cur

def compute_on_dataset(model, data_loader, device, need_confusion, confusion_path, synchronize_gather=True, timer=None):
    # 计算结果
    model.eval() # 将模型状态置于eval, 主要影响 dropout, BN 等操作的行为
    results_dict = {}
    cpu_device = torch.device("cpu")
    torch.cuda.empty_cache()
    # if need_confusion:
    #     print("###########################################################use need_confusion")
    #     confusion_matrix = pickle.load(open(confusion_path + "inverse_confusion.pk", "rb"))
    #     print(confusion_path + "inverse_confusion.pk")
    #     print(confusion_path + "inverse_confusion_ori.pk")
    #     ori_confusion_matrix = pickle.load(open(confusion_path + "inverse_confusion_ori.pk", "rb"))
    to_save = []
    for _, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad(): # 使用model运算时, 不用计算梯度
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                # relation detection needs the targets
                xx, output = model(images.to(device), targets)  # 将计算结果转移到cpu上
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
            cur_data = modify_logits(targets[0], output[0])
            to_save.append(cur_data)
            # if need_confusion:
            #     # print('use need_confusion')
            #     for oi in range(len(output)):
            #         output_label = output[oi].extra_fields['labels']
            #         for pari_idxs, pairs in enumerate(output[oi].extra_fields['rel_pair_idxs']):
            #             pred_rel = output[oi].extra_fields['pred_rel_labels'][pari_idxs].item()
            #             r_name = (output_label[pairs[0].item()].item(), output_label[pairs[1].item()].item(), pred_rel)
            #             need_matrix = torch.zeros(50)
            #             flag_matrix = False
            #             if r_name in confusion_matrix:
            #                 need_matrix += confusion_matrix[r_name]['rel']
            #                 flag_matrix = True
            #             if r_name in ori_confusion_matrix:
            #                 need_matrix += ori_confusion_matrix[r_name]['rel']
            #                 flag_matrix = True
            #             if flag_matrix:
            #                 output[oi].extra_fields['pred_rel_scores'][pari_idxs][1:] *= need_matrix
        if synchronize_gather:
            synchronize()
            multi_gpu_predictions = all_gather({img_id: result for img_id, result in zip(image_ids, output)})
            if is_main_process():
                for p in multi_gpu_predictions:
                    results_dict.update(p)
        else:
            # 更新结果字典
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
        # if _ == 1000:
        #     break
    pickle.dump(to_save, open("tmp/" + 'motifs_test' + ".pk", "wb"))
    print(len(to_save))
    torch.cuda.empty_cache()
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, synchronize_gather=True):
    # 累积预测
    if not synchronize_gather:
        all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return None, None

    if synchronize_gather:
        predictions = predictions_per_gpu
    else:
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
    
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if type(image_ids[-1]) is int and len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!"
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return image_ids, predictions


def inference(
        cfg,
        model,  # 从 build_detection_model 函数得到的模型对象
        data_loader, # PyTorch 的 DataLoader 对象, 对应自定义的数据集
        dataset_name, # str, 数据集的名字
        iou_types=("bbox",), # iou的类型, 默认为 bbox
        box_only=False, # cfg.MODEL.RPN_ONLY="False"
        device="cuda", # cfg.MODEL.DEVICE="cuda"
        expected_results=(), # cfg.TEST.EXPECTED_RESULTS=[]
        expected_results_sigma_tol=4, # cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL=4
        output_folder=None, # 自定义输出文件夹
        logger=None,
):# 模型测试/推演核心代码
    load_prediction_from_cache = cfg.TEST.ALLOW_LOAD_FROM_CACHE and output_folder is not None and os.path.exists(os.path.join(output_folder, "eval_results.pytorch"))
    # convert to a torch.device for efficiency
    # 获取设备
    device = torch.device(device)
    num_devices = get_world_size()
    if logger is None:
        # 日志信息
        logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset # 自定义的数据集类, 如 coco.COCODataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    # 开始计时
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    if load_prediction_from_cache:
        predictions = torch.load(os.path.join(output_folder, "eval_results.pytorch"), map_location=torch.device("cpu"))['predictions']
    else:
        # 调用本文件的函数, 获得预测结果, 关于该函数的解析可看后文
        predictions = compute_on_dataset(model, data_loader, device, cfg.Confusion, cfg.Confusion_Path,synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER, timer=inference_timer)
    # wait for all processes to complete before measuring the time
    # 调用下面的语句, 使得等到所有的进程都结束以后再计算总耗时
    synchronize()
    # 计算总耗时记入log
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    if not load_prediction_from_cache:
        # 调用本文件的函数, 将所有GPU设备上的预测结果累加并返回
        image_ids, predictions = _accumulate_predictions_from_multiple_gpus(predictions, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER)

    if not is_main_process():
        return -1.0

    #if output_folder is not None and not load_prediction_from_cache:
    #    torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    if cfg.TEST.CUSTUM_EVAL:
        detected_sgg = custom_sgg_post_precessing(image_ids, predictions)
        # with open(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_prediction.json'), 'w') as outfile:
        #     json.dump(detected_sgg, outfile)
        import pickle
        with open(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_prediction.pk'), 'wb') as outfile:
            pickle.dump(detected_sgg, outfile)
        print('=====> ' + str(os.path.join(cfg.DETECTED_SGG_DIR, 'custom_prediction.pk')) + ' SAVED !')
        return -1.0

    # 调用评价函数, 返回预测结果的质量
    return evaluate(cfg=cfg,
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    logger=logger,
                    **extra_args)



def custom_sgg_post_precessing(image_ids, predictions):
    output_dict = {}
    for idx, boxlist in zip(image_ids, predictions):
        xyxy_bbox = boxlist.convert('xyxy').bbox
        # current sgg info
        current_dict = {}
        # sort bbox based on confidence
        sortedid, id2sorted = get_sorted_bbox_mapping(boxlist.get_field('pred_scores').tolist())
        # sorted bbox label and score
        bbox = []
        bbox_labels = []
        bbox_scores = []
        # bbox = xyxy_bbox
        # bbox_labels = boxlist.get_field('pred_labels')
        # bbox_scores = boxlist.get_field('pred_scores')
        for i in sortedid:
            bbox.append(xyxy_bbox[i].tolist())
            bbox_labels.append(boxlist.get_field('pred_labels')[i].item())
            bbox_scores.append(boxlist.get_field('pred_scores')[i].item())
        current_dict['bbox'] = torch.tensor(bbox)
        current_dict['bbox_labels'] = torch.tensor(bbox_labels)
        current_dict['bbox_scores'] = torch.tensor(bbox_scores)
        # sorted relationships
        rel_sortedid, _ = get_sorted_bbox_mapping(boxlist.get_field('pred_rel_scores')[:,1:].max(1)[0].tolist())
        # sorted rel
        rel_pairs = []
        rel_labels = []
        rel_scores = []
        rel_all_scores = []
        # rel_labels = boxlist.get_field('pred_rel_scores')[:, 1:].max(1)[1] + 1
        # rel_scores = boxlist.get_field('pred_rel_scores')[:, 1:].max(1)[0]
        # rel_all_scores = boxlist.get_field('pred_rel_scores')
        # id2sorted = torch.tensor(id2sorted, dtype=torch.int32)
        # rel_pairs = [id2sorted[boxlist.get_field('rel_pair_idxs')[:, 0]], id2sorted[boxlist.get_field('rel_pair_idxs')[:, 1]]]
        for i in rel_sortedid:
            rel_labels.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[1].item() + 1)
            rel_scores.append(boxlist.get_field('pred_rel_scores')[i][1:].max(0)[0].item())
            rel_all_scores.append(boxlist.get_field('pred_rel_scores')[i].tolist())
            old_pair = boxlist.get_field('rel_pair_idxs')[i].tolist()
            rel_pairs.append([id2sorted[old_pair[0]], id2sorted[old_pair[1]]])
        current_dict['rel_pairs'] = torch.tensor(rel_pairs)
        current_dict['rel_labels'] = torch.tensor(rel_labels)
        current_dict['rel_scores'] = torch.tensor(rel_scores)
        # current_dict['rel_all_scores'] = torch.tensor(rel_all_scores)
        output_dict[idx] = current_dict
    return output_dict
    
def get_sorted_bbox_mapping(score_list):
    sorted_scoreidx = sorted([(s, i) for i, s in enumerate(score_list)], reverse=True)
    sorted2id = [item[1] for item in sorted_scoreidx]
    id2sorted = [item[1] for item in sorted([(j,i) for i, j in enumerate(sorted2id)])]
    return sorted2id, id2sorted