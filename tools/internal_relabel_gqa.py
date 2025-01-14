# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import json
import time
import datetime
import pickle
import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import numpy as np
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
import json
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed, logger):
    debug_print(logger, 'prepare training')
    model = build_detection_model(cfg)
    debug_print(logger, 'end model construction')

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)

    fix_eval_modules(eval_modules)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor", ]
    else:
        slow_heads = []

    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor"}

    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping["roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch))
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')
    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False,
                                                  update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)
    else:
        # load_mapping is only used when we init current model from detection model.
        checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
    debug_print(logger, 'end load checkpointer')

    if os.path.exists("tmp/" + str(local_rank) + ".json"):
        with open("tmp/" + str(local_rank) + ".json", 'r') as file:
            dic = json.load(file)
        with open("tmp/" + str(local_rank) + ".pk", 'rb') as pkfile:
            to_save = pickle.load(pkfile)
    arguments["iteration"] += (len(dic) / 16 - 2)
    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    debug_print(logger, 'end dataloader')

    logger.info("Start training")
    # dic = {}
    # to_save = []
    end = False
    for iteration, (images, targets, _, train_data) in tqdm(enumerate(train_data_loader)):
        model.eval()
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            pairwise_logits, relation_logits = model(images, targets)
        for t, logits in zip(train_data, relation_logits):
            cur_data = t.extra_fields
            index_pair = []
            for sub_idx, obj_idx, rel in cur_data['relations']:
                index_result = torch.where(torch.all(logits.extra_fields['rel_pair_idxs'].cpu() == torch.tensor([sub_idx, obj_idx]), dim=1))[0]
                if len(index_result) == 0:
                    index_pair.append(0)
                    print(index_result)
                elif len(index_result) == 1:
                    index_pair.append(int(index_result.item()))
                else:
                    print(index_result)
                    index_pair.append(int(index_result[0].item()))
            cur_data = modify_logits(cur_data, logits.extra_fields['pred_rel_scores'][index_pair])
            cur_data["boxes"] = cur_data["boxes"].cpu().numpy()
            cur_data["labels"] = cur_data["labels"].cpu().numpy()
            img_path = cur_data['img_path']
            if img_path in dic:
                continue
            else:
                to_save.append(cur_data)
                dic[img_path] = None
        if len(dic) == len(train_data_loader.dataset.filenames):
            json.dump(dic, open("tmp/" + str(local_rank) + ".json", "w"))
            pickle.dump(to_save, open("tmp/" + str(local_rank) + ".pk", "wb"))
            logger.info(len(dic))
            break
        if iteration % 100 == 0:
            json.dump(dic, open("tmp/" + str(local_rank) + ".json", "w"))
            pickle.dump(to_save, open("tmp/" + str(local_rank) + ".pk", "wb"))
            logger.info(len(dic))

    return train_data_loader.dataset.filenames


def modify_logits(cur_data, rel_logits):
    # just append the logits
    cur_data["logits"] = rel_logits.cpu().numpy()
    return cur_data


def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False


def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    # print(args.local_rank)
    file_names = train(cfg, args.local_rank, args.distributed, logger)
    # synchronize()
    # if dist.is_available() and dist.get_rank() == 0:
    l = []
    for r in range(num_gpus):
        s = pickle.load(open("tmp/" + str(r) + ".pk", "rb"))
        l.extend(s)
    dic = {}
    for d in tqdm(l):
        dic[d['img_path']] = d
    rst_l = [dic['datasets/gqa/images/'+k] for k in file_names]
    save_path = os.path.join(output_dir, "motif_GQA_confusion.pk")
    # save_path = os.path.join(output_dir, "em_confusion.pk")
    pickle.dump(rst_l, open(save_path, "wb"))


if __name__ == "__main__":
    main()
