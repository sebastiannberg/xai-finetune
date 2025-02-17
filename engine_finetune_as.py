# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import numpy as np
import torch

from timm.data import Mixup
from timm.utils import accuracy

# import util.misc as misc
# import util.lr_sched as lr_sched
# from util.stat import calculate_stats, concat_all_gather
import models.AudioMAE.util.misc as misc
import models.AudioMAE.util.lr_sched as lr_sched
from models.AudioMAE.util.stat import calculate_stats, concat_all_gather



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, _vids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, dist_eval=False):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    outputs=[]
    targets=[]
    vids=[]
    for batch in metric_logger.log_every(data_loader, 300, header):

        images = batch[0]
        target = batch[1]
        vid = batch[2]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            # remark: 
            # 1. use concat_all_gather and --dist_eval for faster eval by distributed load over gpus
            # 2. otherwise comment concat_all_gather and remove --dist_eval one every gpu
            if dist_eval:
                output = concat_all_gather(output)
                target = concat_all_gather(target)
            outputs.append(output)
            targets.append(target)
            vids.append(vid)

    outputs=torch.cat(outputs).cpu().numpy()
    targets=torch.cat(targets).cpu().numpy()
    vids = [j for sub in vids for j in sub]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    probabilities = sigmoid(outputs)

    threshold = 0.1
    predicted_labels = probabilities > threshold

    name_dict = data_loader.dataset.name_dict

    for i in range(len(outputs)):
        print(f"Video ID: {vids[i]}")
        predicted_label_indices = np.where(predicted_labels[i] == 1)[0]
        predicted_label_names = [name_dict[str(index)] for index in predicted_label_indices]
        print("Predicted:", predicted_label_names)
        predicted_target_indices = np.where(targets[i] == 1)[0]
        target_names = [name_dict[str(index)] for index in predicted_target_indices]
        print("Target:", target_names)

    # stats = calculate_stats(outputs, targets)
    # AP = [stat['AP'] for stat in stats]
    # mAP = np.mean([stat['AP'] for stat in stats])
    # print("mAP: {:.6f}".format(mAP))
    return {"mAP": None, "AP": None}


