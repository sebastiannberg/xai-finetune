import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from timm.models.layers import trunc_normal_
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import datetime
import random
import os
from collections import defaultdict
import matplotlib.pyplot as plt

import models_vit as models_vit

def perform_analysis(model, device, criterion, class_attention_grads, args, epoch, data_loader_stats, data_loader_plot):
    for fbank, label in data_loader_stats:
        fbank = fbank.to(device)
        label = label.to(device)

        logits, attention = model(fbank, return_attention=True)

        classification_loss = criterion(logits, label)
        logger.info(f'classification loss: {args.alpha * classification_loss.item()}')

        label_indices = torch.argmax(label, dim=1)
        selected_attention_grads = class_attention_grads[label_indices, ...]
        selected_attention_grads = args.grad_scale * selected_attention_grads

        # attention_wo_cls = attention[:, :, :, 1:, 1:]
        # attention_wo_cls_softmaxed = attention_wo_cls.softmax(dim=-1)
        # grads_wo_cls = selected_attention_grads[:, :, :, 1:, 1:]

        pre_attention_interpret = attention * selected_attention_grads
        post_attention_interpret = pre_attention_interpret.softmax(dim=-1)
        interpret_loss = -(post_attention_interpret.detach() * (attention + 1e-12).log()).sum(dim=-1).mean()
        logger.info(f'interpret loss: {(1 - args.alpha) * interpret_loss.item()}')

        # logger.info(
        #     f"Attention stats:\n"
        #     f"  shape: {attention.shape}\n"
        #     f"  max: {attention.max().item()}\n"
        #     f"  min: {attention.min().item()}\n"
        #     f"  mean: {attention.mean().item()}\n"
        #     f"  std: {attention.std().item()}"
        # )
        # logger.info(
        #     f"Attention wo CLS stats:\n"
        #     f"  shape: {attention_wo_cls.shape}\n"
        #     f"  max: {attention_wo_cls.max().item()}\n"
        #     f"  min: {attention_wo_cls.min().item()}\n"
        #     f"  mean: {attention_wo_cls.mean().item()}\n"
        #     f"  std: {attention_wo_cls.std().item()}"
        # )
        # logger.info(
        #     f"Attention wo CLS softmaxed stats:\n"
        #     f"  shape: {attention_wo_cls_softmaxed.shape}\n"
        #     f"  max: {attention_wo_cls_softmaxed.max().item()}\n"
        #     f"  min: {attention_wo_cls_softmaxed.min().item()}\n"
        #     f"  mean: {attention_wo_cls_softmaxed.mean().item()}\n"
        #     f"  std: {attention_wo_cls_softmaxed.std().item()}"
        # )
        # logger.info(
        #     f"Selected Attention Grads stats:\n"
        #     f"  shape: {selected_attention_grads.shape}\n"
        #     f"  max: {selected_attention_grads.max().item()}\n"
        #     f"  min: {selected_attention_grads.min().item()}\n"
        #     f"  mean: {selected_attention_grads.mean().item()}\n"
        #     f"  std: {selected_attention_grads.std().item()}"
        # )
        # logger.info(
        #     f"Selected Attention Grads wo CLS stats:\n"
        #     f"  shape: {grads_wo_cls.shape}\n"
        #     f"  max: {grads_wo_cls.max().item()}\n"
        #     f"  min: {grads_wo_cls.min().item()}\n"
        #     f"  mean: {grads_wo_cls.mean().item()}\n"
        #     f"  std: {grads_wo_cls.std().item()}"
        # )
        # logger.info(
        #     f"Pre-Softmax Attention Interpret stats:\n"
        #     f"  shape: {pre_attention_interpret.shape}\n"
        #     f"  max: {pre_attention_interpret.max().item()}\n"
        #     f"  min: {pre_attention_interpret.min().item()}\n"
        #     f"  mean: {pre_attention_interpret.mean().item()}\n"
        #     f"  std: {pre_attention_interpret.std().item()}"
        # )
        # logger.info(
        #     f"Post-Softmax Attention Interpret stats:\n"
        #     f"  shape: {post_attention_interpret.shape}\n"
        #     f"  max: {post_attention_interpret.max().item()}\n"
        #     f"  min: {post_attention_interpret.min().item()}\n"
        #     f"  mean: {post_attention_interpret.mean().item()}\n"
        #     f"  std: {post_attention_interpret.std().item()}"
        # )

        heatmap_fig = plot_attention_heatmap(attention, title="Attention Heatmap")
        heatmap_fig.savefig(os.path.join(epoch_path, 'attention_heatmap.png'))
        plt.close(heatmap_fig)

        grads_heatmap_fig = plot_attention_heatmap(class_attention_grads, title="Attentiong Gradient Heatmap")
        grads_heatmap_fig.savefig(os.path.join(epoch_path, 'class_attention_grads_heatmap.png'))
        plt.close(grads_heatmap_fig)

        attention_interpret_heatmap_fig = plot_attention_heatmap(post_attention_interpret, title="Attention Interpret Heatmap")
        attention_interpret_heatmap_fig.savefig(os.path.join(epoch_path, 'post_attention_interpret_heatmap.png'))
        plt.close(attention_interpret_heatmap_fig)

        attention_cls_argmax_percentage = cls_argmax_percentage(attention)
        logger.info(f"Attention Argmax Percentage for [CLS] token: {attention_cls_argmax_percentage}")

        avg_received_attn_cls = avg_received_attention_cls(attention)
        logger.info(f"Average Received Attention for [CLS] token: {avg_received_attn_cls}")

def main():
    dataset_train = UrbanDataset(
        root=URBAN_PATH,
        fold=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        mixup_prob=0.5,
        roll_mag_aug=True,
        target_length=args.target_length,
        freqm=48,
        timem=192,
        num_classes=args.num_classes
    )
    dataset_val = UrbanDataset(
        root=URBAN_PATH,
        fold=[10],
        mixup_prob=0.0,
        roll_mag_aug=False,
        target_length=args.target_length,
        freqm=0,
        timem=0,
        num_classes=args.num_classes
    )
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    # Partition data to samples for each label
    dataset_interpret = UrbanDataset(
        root=URBAN_PATH,
        fold=[1], # TODO: Try all training folds
        mixup_prob=0.0,
        roll_mag_aug=False,
        target_length=args.target_length,
        freqm=0,
        timem=0,
        num_classes=args.num_classes
    )
    class_to_indices = defaultdict(list)
    for idx in range(len(dataset_interpret)):
        _, data_y = dataset_interpret[idx]
        class_idx = data_y.argmax().item()
        class_to_indices[class_idx].append(idx)
    class_loaders = []
    for class_idx in range(args.num_classes):
        indices = class_to_indices[class_idx]
        subset = Subset(dataset_interpret, indices)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g
        )
        class_loaders.append(loader)

