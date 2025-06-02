import argparse
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from einops import rearrange
from typing import Union
from tqdm import tqdm

from urban_dataset import UrbanDataset
from models_vit import PatchEmbed_new
import models_vit as models_vit


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "UrbanSound8K")
RESULT_PATH = os.path.join(PROJECT_ROOT, "deletion_insertion_results")
CKPT_PATHS = {
    "baseline_with_augmentations_lr10-5_wd10-5_bs8": "/home/sebastian/dev/xai-finetune/results/final_experiments/baseline_with_augmentations_lr10-5_wd10-5_bs8/ckpt/epoch_60.pth",
    "baseline_without_augmentations_lr10-5_wd10-5_bs8": "/home/sebastian/dev/xai-finetune/results/final_experiments/baseline_without_augmentations_lr10-5_wd10-5_bs8/ckpt/epoch_60.pth",
    "ifi_a90_t10-5_se2_wball": "/home/sebastian/dev/xai-finetune/results/final_experiments/ifi_a90_t10-5_se2_wball/ckpt/epoch_60.pth",
    "et_se1_k0": "/home/sebastian/dev/xai-finetune/results/final_experiments/et_se1_k0/ckpt/epoch_60.pth",
    "et_se10_k0": "/home/sebastian/dev/xai-finetune/results/final_experiments/et_se10_k0/ckpt/epoch_60.pth",
    "et_with_augmentations_se1_k0": "/home/sebastian/dev/xai-finetune/results/final_experiments/et_with_augmentations_se1_k0/ckpt/epoch_60.pth",
    "et_with_augmentations_se10_k0": "/home/sebastian/dev/xai-finetune/results/final_experiments/et_with_augmentations_se10_k0/ckpt/epoch_60.pth",
}

def get_args():
    parser = argparse.ArgumentParser(description="Deletion-Insertion XAI Benchmark")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of target classes in output")
    parser.add_argument("--target_length", type=int, default=512, help="Number of time frames for fbank spectrograms")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of worker threads for data loading")
    return parser.parse_args()

def attention_rollout(model, attention):
    _, num_layers, _, seq_length, _ = attention.shape
    identity = torch.eye(seq_length, device=attention.device).unsqueeze(0) # [1, S, S]

    # 1) Merge heads by averaging: one [B, S, S] per layer
    merged_per_layer = []
    for layer_idx in range(num_layers):
        # attention[:, layer_idx] is [B, H, S, S]
        avg_attention = attention[:, layer_idx].mean(dim=1) # [B, S, S]
        merged_per_layer.append(avg_attention)

    # 2) Rollout: start at layer 0 without residual
    rollout_map = merged_per_layer[0]

    # 3) For layers 1…L-1, do 0.5A + 0.5I then matmul
    for layer_attention in merged_per_layer[1:]:
        residual_attention = 0.5 * layer_attention + 0.5 * identity # [B, S, S]
        rollout_map = torch.matmul(residual_attention, rollout_map) # [B, S, S]

    patch_scores = rollout_map[:, 0, 1:] # -> (batch, seq-1) only cls row, and drop cls column

    h_patches, w_patches = model.patch_embed.patch_hw
    patch_grid = patch_scores.view(-1, 1, h_patches, w_patches) # -> (batch, 1, h_patches, w_patches)

    h_img, w_img = model.patch_embed.img_size
    saliency_maps = torch.nn.functional.interpolate(patch_grid, size=(h_img, w_img), mode="bilinear") # -> (batch, 1, h_img, w_img)

    saliency_mins = saliency_maps.amin(dim=(-1, -2), keepdim=True)
    saliency_maxs = saliency_maps.amax(dim=(-1, -2), keepdim=True)
    normalized_saliency_maps = (saliency_maps - saliency_mins) / (saliency_maxs - saliency_mins + 1e-10)
    return normalized_saliency_maps

def attention_gradients(model, fbank, label):
    model.zero_grad()
    with torch.autograd.set_grad_enabled(True):
        # Forward
        logits = model(fbank)
        target_logit = logits[0, label.argmax(dim=1)[0].item()]

        # Ensure 'retain_grad()' for each attention block before backprop
        for block in model.blocks:
            if hasattr(block.attn, 'attn') and block.attn.attn is not None:
                if block.attn.attn.requires_grad:
                    block.attn.attn.retain_grad()
                    block.attn.attn_pre_softmax.retain_grad()

        # Backprop
        target_logit.backward()

        # Collect attention grads and plot
        all_grads = []
        for i, block in enumerate(model.blocks):
            if hasattr(block.attn, 'attn') and block.attn.attn.grad is not None:
                all_grads.append(block.attn.attn.grad.detach().clone())

        grads = torch.stack(all_grads, dim=1) # -> (blocks, heads, seq, seq)
        grads = grads.mean(dim=1)

        # Remove retained gradients to clean up
        for block in model.blocks:
            if hasattr(block.attn, 'attn') and block.attn.attn is not None:
                if block.attn.attn.grad is not None:
                    block.attn.attn.grad = None
        model.zero_grad()

    patch_scores = grads[:, -1, 0, 1:] # -> (batch, seq-1) only cls row, and drop cls column, last layer

    h_patches, w_patches = model.patch_embed.patch_hw
    patch_grid = patch_scores.view(-1, 1, h_patches, w_patches) # -> (batch, 1, h_patches, w_patches)

    h_img, w_img = model.patch_embed.img_size
    saliency_maps = torch.nn.functional.interpolate(patch_grid, size=(h_img, w_img), mode="bilinear") # -> (batch, 1, h_img, w_img)

    saliency_mins = saliency_maps.amin(dim=(-1, -2), keepdim=True)
    saliency_maxs = saliency_maps.amax(dim=(-1, -2), keepdim=True)
    normalized_saliency_maps = (saliency_maps - saliency_mins) / (saliency_maxs - saliency_mins + 1e-10)
    return normalized_saliency_maps

def deletion(
    x: torch.Tensor, # batch of images (1 c h w)
    expl: torch.Tensor, # batch of explanations (1 h w)
    bg: torch.Tensor, # channel-wise background values (c,)
    f: torch.nn.Module, # classifier
    labels: torch.Tensor, # label indices to evaluate
    K: Union[float, int] = 0.4, # proportion or number of removed pixels
    step: int = 100, # number of pixels modified per one iteration
    reduction: str = 'mean' # 'sum' or 'mean'
):
    b, c, h, w = x.shape
    assert b == 1
    x0 = x
    x = x.squeeze(0)
    expl = expl.squeeze(0)
    K_ = int(K * h * w) if isinstance(K, float) else K
    maxiter = K_ // step

    x_flat = rearrange(x, 'c h w -> c (h w)')
    expl_flat = rearrange(expl, 'h w -> (h w)')
    bg_ = bg.view(c, 1)

    _, indices = torch.sort(expl_flat, descending=True)

    cls_idx = labels.argmax(dim=-1).item()

    with torch.no_grad():
        orig_logit = f(x0)                                                 # (1, num_classes)
        orig_conf  = torch.softmax(orig_logit, dim=-1)[0, cls_idx]         # scalar
        # make a fully‐background image of same size:
        blank_flat = bg_.repeat(1, x_flat.size(1))
        blank_img  = rearrange(blank_flat, 'c (h w) -> 1 c h w', h=h)
        blank_logit= f(blank_img)
        blank_conf = torch.softmax(blank_logit, dim=-1)[0, cls_idx]        # scalar

    scores = []
    with torch.no_grad():
        for i in range(1, maxiter + 1):
            to_remove = indices[: i * step]
            x_mod = x_flat.clone()
            x_mod[:, to_remove] = bg_
            x_mod = rearrange(x_mod, 'c (h w) -> 1 c h w', h=h)

            conf  = torch.softmax(f(x_mod), dim=-1)[0, cls_idx]

            norm_i = (conf - blank_conf) / (orig_conf - blank_conf)
            norm_i = norm_i.clamp(0.0, 1.0)
            scores.append(norm_i)

    scores = torch.stack(scores, dim=0)

    if reduction == 'sum':
        return scores.sum()
    else:
        return scores.mean()

def insertion(
    x: torch.Tensor, # batch of images (b c h w)
    expl: torch.Tensor, # batch of explanations (b h w)
    bg: torch.Tensor, # channel-wise background values (c,)
    f: torch.nn.Module, # classifier
    labels: torch.Tensor, # label indices to evaluate
    K: Union[float, int] = 0.4, # proportion or number of inserted pixels
    step: int = 100, # number of pixels modified per one iteration
    reduction: str = 'mean' # 'sum' or 'mean'
):
    b, c, h, w = x.shape
    assert b == 1
    x0 = x
    x = x.squeeze(0)
    expl = expl.squeeze(0)
    K_ = int(K * h * w) if isinstance(K, float) else K
    maxiter = K_ // step

    x_flat = rearrange(x, 'c h w -> c (h w)')
    expl_flat = rearrange(expl, 'h w -> (h w)')
    bg_ = bg.view(c, 1)

    _, indices = torch.sort(expl_flat, descending=True)

    cls_idx = labels.argmax(dim=-1).item()

    with torch.no_grad():
        orig_conf  = torch.softmax(f(x0), dim=-1)[0, cls_idx]
        blank_flat = bg_.repeat(1, x_flat.size(1))
        blank_img  = rearrange(blank_flat, 'c (h w) -> 1 c h w', h=h)
        blank_conf = torch.softmax(f(blank_img), dim=-1)[0, cls_idx]

    scores = []
    with torch.no_grad():
        for i in range(0, maxiter+1):
            keep_idx = indices[: i*step]
            x_mod = bg_.repeat(1, x_flat.size(1))
            x_mod[:, keep_idx] = x_flat[:, keep_idx]
            x_mod = rearrange(x_mod, 'c (h w) -> 1 c h w', h=h)

            conf  = torch.softmax(f(x_mod), dim=-1)[0, cls_idx]

            norm_i = (conf - blank_conf) / (orig_conf - blank_conf)
            norm_i = norm_i.clamp(0.0, 1.0)
            scores.append(norm_i)

    scores = torch.stack(scores, dim=0)

    if reduction == 'sum':
        return scores.sum()
    else:
        return scores.mean()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # One random chosen sample for each class from validation set
    filenames = [
        "73524-0-0-5.wav",
        "2937-1-0-0.wav",
        "14470-2-0-10.wav",
        "7965-3-3-0.wav",
        "26344-4-0-0.wav",
        "15544-5-0-0.wav",
        "25037-6-0-0.wav",
        "34050-7-2-0.wav",
        "26173-8-0-0.wav",
        "77901-9-0-2.wav",
    ]
    dataset_val = UrbanDataset(
        root=DATASET_PATH,
        fold=[10],
        mixup_prob=0.0,
        roll_mag_aug=False,
        target_length=args.target_length,
        freqm=0,
        timem=0,
        num_classes=args.num_classes
    )
    selected_idxs = [
        idx
        for idx, (_, _, filename) in enumerate(dataset_val)
        if os.path.basename(filename) in filenames
    ]
    selected_dataset = Subset(dataset_val, selected_idxs)
    data_loader_val = DataLoader(
        selected_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # means = []
    # for fbank, _, _ in data_loader_val:
    #     means.append(fbank.mean(dim=(0,2,3)).squeeze().item())
    # mean_bg = np.mean(means)
    # print(mean_bg)
    # raise ValueError

    results = {}
    for name, ckpt_path in CKPT_PATHS.items():
        print(f"Evaluating {name} @ {ckpt_path}")

        model = models_vit.__dict__['vit_base_patch16'](
            num_classes=args.num_classes,
            drop_path_rate=0.1,
            global_pool=True,
            mask_2d=True,
            use_custom_patch=False
        )
        model.patch_embed = PatchEmbed_new(
            img_size=(args.target_length, 128),
            patch_size=(16, 16),
            in_chans=1,
            embed_dim=768,
            stride=16
        )
        model.pos_embed = nn.Parameter(
            torch.zeros(1, model.patch_embed.num_patches + 1, 768),
            requires_grad=False
        )
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint_state_dict = checkpoint['model_state_dict']
        with torch.no_grad():
            model.patch_embed.proj.weight.copy_(checkpoint_state_dict['patch_embed.proj.weight'])
            model.patch_embed.proj.bias.copy_(checkpoint_state_dict['patch_embed.proj.bias'])
            model.pos_embed.data.copy_(checkpoint_state_dict['pos_embed'])
        model.load_state_dict(checkpoint_state_dict, strict=False)
        model.to(device)
        print(f'Device: {device}')


        del_rollout_list = []
        ins_rollout_list = []
        del_grad_list    = []
        ins_grad_list    = []
        model.eval()
        for fbank, label, _ in tqdm(data_loader_val, total=len(data_loader_val)):
            fbank = fbank.to(device) # (batch, 1, time, freq) batch is 1
            label = label.to(device)

            _, attention = model(fbank, return_attention=True)

            saliency_rollout = attention_rollout(model, attention) # -> (batch, 1, time, freq)
            del_scores_rollout = deletion(fbank, saliency_rollout.squeeze(1), torch.tensor(data=0.02).to(device), model, label)
            ins_scores_rollout = insertion(fbank, saliency_rollout.squeeze(1), torch.tensor(data=0.02).to(device), model, label)

            saliency_grad = attention_gradients(model, fbank, label) # -> (batch, 1, time, freq)
            del_scores_grad = deletion(fbank, saliency_grad.squeeze(1), torch.tensor(data=0.02).to(device), model, label)
            ins_scores_grad = insertion(fbank, saliency_grad.squeeze(1), torch.tensor(data=0.02).to(device), model, label)

            del_rollout_list.append(del_scores_rollout.item())
            ins_rollout_list.append(ins_scores_rollout.item())
            del_grad_list.append(del_scores_grad.item())
            ins_grad_list.append(ins_scores_grad.item())

        # compute “official” deletion/insertion scores: mean over steps for each sample, then mean over samples
        del_rollout_score = float(np.mean(del_rollout_list))
        ins_rollout_score = float(np.mean(ins_rollout_list))
        del_grad_score = float(np.mean(del_grad_list))
        ins_grad_score = float(np.mean(ins_grad_list))

        results[name] = {
            "deletion_rollout_score": del_rollout_score,
            "insertion_rollout_score": ins_rollout_score,
            "deletion_gradients_score": del_grad_score,
            "insertion_gradients_score": ins_grad_score,
        }
        for key, value in results[name].items():
            print(key)
            print(value)

    df = pd.DataFrame(results).T
    df = df[[
        "deletion_rollout_score",
        "insertion_rollout_score",
        "deletion_gradients_score",
        "insertion_gradients_score"
    ]]
    print("\n=== Deletion/Insertion Scores ===")
    print(df.to_markdown(floatfmt=".4f"))

    # Save results as json
    os.makedirs(RESULT_PATH, exist_ok=True)
    json_path = os.path.join(RESULT_PATH, "deletion_insertion_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results -> {json_path}")

    # TODO: 4 plots one for each score
    # Plot model comparisons
    ax = df.plot(
        kind="bar",
        figsize=(10, 6),
        rot=0,
        title="Deletion/Insertion Score Comparison"
    )
    ax.set_ylabel("Score")
    plt.tight_layout()
    plot_path = os.path.join(RESULT_PATH, "model_comparison.png")
    plt.savefig(plot_path, dpi=250)
    print(f"\nSaved comparison plot -> {plot_path}")

if __name__ == "__main__":
    main()
    print("done")
