# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# 常规包
import datetime
import logging
import time

import torch
import torch.distributed as dist # 分布式相关

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
     对loss进行reduce, 使其可以利用 rank 0进行处理
    """
    world_size = get_world_size()
    if world_size < 2: # 单 GPU, 直接返回, 无需reduce
        return loss_dict
    with torch.no_grad(): # 不要计算任何参数的梯度
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k) # 获取键
            all_losses.append(loss_dict[k]) # 获取值
        # 将列表中的 loss 连接起来组成一个一维的tensor, tensor的每个元素代表一个 loss.
        all_losses = torch.stack(all_losses, dim=0)
        # import torch.distributed as dist
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses
