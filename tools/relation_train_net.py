# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
# 导入各种包
# 下面的这个必须在导入所有包之前导入, 不能放到其他位置. TODO 原因
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
# 常规包
import argparse
import os
import time
import datetime

import torch
from torch.nn.utils import clip_grad_norm_

from maskrcnn_benchmark.config import cfg # 导入默认配置信息
from maskrcnn_benchmark.data import make_data_loader # 数据集载入
from maskrcnn_benchmark.solver import make_lr_scheduler # 学习率更新策略
from maskrcnn_benchmark.solver import make_optimizer # 设置优化器, 封装了PyTorch的SGD类
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference # 推演代码
# 调用了 ./maskrcnn_benchmark/modeling/detector/ 中的 build_detection_model() 函数
# 该函数和 Detectron 中的类似, 都是用来创建目标检测模型的, 这也是创建模型的入口函数, 十分重要
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
# 封装了 PyTorch 的 torch.utils.collect_env.get_preety_env_info 函数, 同时附加了 PIL.__version__ 版本新
from maskrcnn_benchmark.utils.collect_env import collect_env_info
# 分布式训练相关设置, 由于我的gpu个数为1, 因此 get_rank() 会返回 0
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
# 封装了 logging 模块, 用于向屏幕输出一些日志信息
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
# 封装了 os.mkdirs 函数, 当文件夹已存在时会自动略过, 不会提示错误
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger


# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

# ... 训练脚本核心代码
def train(cfg, local_rank, distributed, logger):
    """
    根据配置文件来创建模型 (发现cfg被传到了build_detection_model()函数中)
    """
    debug_print(logger, 'prepare training')
    # 该语句调用了 ./maskrcnn_benchmark/modeling/detector/ 中的 build_detection_model() 函数
    # 该函数和 Detectron 中的类似, 都是用来创建目标检测模型的, 这也是创建模型的入口函数, 十分重要
    # 这里我们只需要知道该函数会根据我们的配置文件返回一个网络模型就可以了
    model = build_detection_model(cfg)
    print(model)
    debug_print(logger, 'end model construction')

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
    fix_eval_modules(eval_modules)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor",]
    else:
        slow_heads = []

    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor" : "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor" : "roi_heads.box.feature_extractor"}
    
    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping["roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    # 默认为 "cuda"
    device = torch.device(cfg.MODEL.DEVICE)
    # 将模型移到指定设备上
    model.to(device)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH #12
    # 封装了 torch.optiom.SGD() 函数, 根据tensor的requires_grad属性构成需要更新的参数列表
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch))
    # 根据配置信息设置 optimizer 的学习率更新策略
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    # 分布式训练情况下, 并行处理数据
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')
    # 创建一个参数字典, 并将迭代次数置为0
    arguments = {}
    arguments["iteration"] = 0

    # 获取输出的文件夹路径, 默认为 '.', 这里建议将 cfg.OUTPUT_DIR 提前设置成自己的路径.
    output_dir = cfg.OUTPUT_DIR

    # 因为我只有一个gpu, 所以这里 save_to_disk=True
    save_to_disk = get_rank() == 0
    # DetectronCheckpointer 对象, 后面会用在 do_train() 函数的参数
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if checkpointer.has_checkpoint():
        # 加载指定权重文件
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False,
                                       update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        # 字典的update方法, 对字典的键值进行更新
        arguments.update(extra_checkpoint_data)
    else:
        # load_mapping is only used when we init current model from detection model.
        checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
    debug_print(logger, 'end load checkpointer')
    # data_loader 的类型为列表, 内部元素类型为 torch.utils.data.DataLoader
    # 注意, 当mode=train时, 要确保 cfg.DATASETS.TRAIN 的值为一个列表
    # 并且必须指向一个或多个存在的anns文件, 默认情况下该值为空, 所以必须在配置文件中指定
    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=distributed,
    )
    debug_print(logger, 'end dataloader')
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, distributed, logger)

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    print_first_grad = True
    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        model.train()
        fix_eval_modules(eval_modules)

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()

        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad # print grad or not
        print_first_grad = False
        clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad], max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)

        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 200 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[-1]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if iteration % checkpoint_period == 0 or iteration == 1:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        val_result = None # used for scheduler updating
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            logger.info("Start validating")
            val_result = run_val(cfg, model, val_data_loaders, distributed, logger)
            logger.info("Validation Result: %.4f" % val_result)
 
        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            scheduler.step(val_result, epoch=iteration)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                break
        else:
            scheduler.step()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False,
                                              update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
    return model

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False

def run_val(cfg, model, val_data_loaders, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    # iou_types = ("bbox",)
    iou_types = ()
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
                            cfg,
                            model,
                            val_data_loader,
                            dataset_name=dataset_name,
                            iou_types=iou_types,
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=None,
                            logger=logger,
                        )
        synchronize()
        val_result.append(dataset_result)
    # support for multi gpu distributed testing
    gathered_result = all_gather(torch.tensor(dataset_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result>=0]
    val_result = float(valid_result.mean())
    del gathered_result, valid_result
    torch.cuda.empty_cache()
    return val_result

# ... 推演代码
def run_test(cfg, model, distributed, logger):
    if distributed:
        model = model.module
    # TODO check if it helps. releases unoccupied memory
    torch.cuda.empty_cache()
    iou_types = ()
    # iou_types = ("bbox",)
    # 如果mask为Ture, 则添加分割信息
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )
    # 根据标签文件数确定输出文件夹数
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names): # 遍历标签文件
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder) # 创建输出文件夹
            output_folders[idx] = output_folder  # 将文件夹的路径名放入列表
    # 根据配置文件信息创建数据集
    data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
    # 遍历每个标签文件, 执行 inference 过程.
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
        )
        synchronize() # 多GPU推演时的同步函数

# ... 主函数, 会在内部调用上述函数
def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    # 配置文件(带有.yaml后缀的配置文件作为命令行参数输入)
    parser.add_argument(
        # 这里虽然是 config-file, 但要用 args.config_file来访问, 不能用 args.config-file 访问.
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
        nargs=argparse.REMAINDER,# 这一行不能少
    )

    args = parser.parse_args()

    # 获取 gpu 的数量, 我电脑只有一个GPU, 故 num_gpus = 1
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # 是否进行多GPU的分布式训练
    args.distributed = num_gpus > 1

    # 设置分布式训练时的一些初始化信息
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    # 将配置文件进行读取
    # 将 config_file 指定的配置项覆盖到默认配置项当中
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()# 冻结所有的配置项, 防止修改

    output_dir = cfg.OUTPUT_DIR
    if output_dir: # 这里封装了 os.mkdir, 使得当output_dir存在时, 会直接略过, 不会返回错误
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    # 打开指定的配置文件, 并读取其中的相关信息, 将值存储在 config_str 中, 然后输出到屏幕上
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)
    # 将配置对象 cfg 作为参数传入train()函数
    # 调用 train 函数, 该函数会执行模型训练的代码逻辑, 详解可看后文
    model = train(cfg, args.local_rank, args.distributed, logger)

    if not args.skip_test:
        # 同理, 该函数会执行模型推演的代码逻辑, 详情可看后文
        run_test(cfg, model, args.distributed, logger)


if __name__ == "__main__":
    main()
