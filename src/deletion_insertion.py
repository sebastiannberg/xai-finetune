import argparse
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from urban_dataset import UrbanDataset
from models_vit import PatchEmbed_new
import models_vit as models_vit


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "UrbanSound8K")
RESULT_PATH = os.path.join(PROJECT_ROOT, "deletion_insertion_results")
CKPT_PATHS = {
    "baseline_with_augmentations_lr10-5_wd10-5_bs8": "/home/sebastian/dev/xai-finetune/results/final_experiments/baseline_with_augmentations_lr10-5_wd10-5_bs8/ckpt/epoch_60.pth",
    "et_with_augmentations_se10_k0": "/home/sebastian/dev/xai-finetune/results/final_experiments/et_with_augmentations_se10_k0/ckpt/epoch_60.pth"
}

def get_args():
    parser = argparse.ArgumentParser(description="Deletion-Insertion XAI Benchmark")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of target classes in output")
    parser.add_argument("--target_length", type=int, default=512, help="Number of time frames for fbank spectrograms")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of worker threads for data loading")
    # parser.add_argument("--temperature", type=float, default=1e-5, help="Scaling up gradients to avoid uniform distribution for attention_interpret")
    # parser.add_argument("--sigma_k", type=float, default=1.0, help="K-value for K-sigma thresholding")
    return parser.parse_args()

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
        for fbank, label, _ in data_loader_val:
            fbank = fbank.to(device) # (batch, 1, time, freq) batch is 1
            label = label.to(device)

            _, attention = model(fbank, return_attention=True)

            saliency_rollout = attention_rollout(model, attention) # -> (batch, 1, time, freq)
            del_curve_rollout = compute_deletion(fbank, saliency_rollout, model, label) # -> (batch, n_steps)
            ins_curve_rollout = compute_insertion(fbank, saliency_rollout, model, label) # -> (batch, n_steps)

            saliency_grad = attention_gradients(model, fbank, label) # -> (batch, 1, time, freq)
            del_curve_grad = compute_deletion(fbank, saliency_grad, model, label) # -> (batch, n_steps)
            ins_curve_grad = compute_insertion(fbank, saliency_grad, model, label) # -> (batch, n_steps)

            del_rollout_list.append(del_curve_rollout)
            ins_rollout_list.append(ins_curve_rollout)
            del_grad_list.append(del_curve_grad)
            ins_grad_list.append(ins_curve_grad)

        # Concatenate across all batches -> (n_samples, n_steps)
        del_r = torch.cat(del_rollout_list, dim=0)
        ins_r = torch.cat(ins_rollout_list, dim=0)
        del_g = torch.cat(del_grad_list, dim=0)
        ins_g = torch.cat(ins_grad_list, dim=0)

        # Compute per‐sample AUC via trapezoidal rule, then average
        auc_del_r = np.trapz(del_r.cpu().numpy(), axis=1).mean()
        auc_ins_r = np.trapz(ins_r.cpu().numpy(), axis=1).mean()
        auc_del_g = np.trapz(del_g.cpu().numpy(), axis=1).mean()
        auc_ins_g = np.trapz(ins_g.cpu().numpy(), axis=1).mean()

        results[name] = {
            "deletion_rollout_auc": float(auc_del_r),
            "insertion_rollout_auc": float(auc_ins_r),
            "deletion_gradients_auc": float(auc_del_g),
            "insertion_gradients_auc": float(auc_ins_g),
        }

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

def compute_deletion(inputs, saliency, model, label, n_steps=50):
    """
    inputs: (1, C, time, freq) original input
    saliency: (1, 1, time, freq) normalized in [0,1]
    model: nn.Module
    label: one-hot (1, N)
    returns: (1, n_steps) deletion curve
    """
    C, time, freq = inputs.shape[1:]
    flat_saliency = saliency.view(-1)
    order = torch.argsort(flat_saliency, descending=True) # highest saliency first

    # how many pixels to remove per step
    total = time*freq
    step  = max(1, total // n_steps)
    mask = torch.ones_like(flat_saliency)

    curve = []
    for i in range(0, total, step):
        # zero out the top i pixels
        mask[order[:i]] = 0 

        x_flat = inputs.view(1, C, -1)
        x_mask = x_flat * mask
        x_mask = x_mask.view(1, C, time, freq)

        # forward
        logits = model(x_mask)
        probs = F.softmax(logits, dim=1)
        idx = label.argmax(dim=1)[0].item()
        curve.append(probs[0, idx].item())

    # pad/truncate to exactly n_steps
    if len(curve) < n_steps:
        curve += [curve[-1]] * (n_steps-len(curve))
    curve = torch.tensor(curve[:n_steps]).unsqueeze(0)
    return curve

def compute_insertion(inputs, saliency, model, label, n_steps=50):
    """
    start from zeros and gradually insert pixels
    """
    C, time, freq = inputs.shape[1:]
    flat_saliency = saliency.view(-1)
    order = torch.argsort(flat_saliency, descending=True)

    total = time*freq
    step = max(1, total // n_steps)
    mask = torch.zeros_like(flat_saliency)

    curve = []
    for i in range(0, total, step):
        mask[order[:i]] = 1 
        x_flat = inputs.view(1, C, -1)
        x_ins = x_flat * mask
        x_ins = x_ins.view(1, C, time, freq)

        logits = model(x_ins)
        probs = F.softmax(logits, dim=1)
        idx = label.argmax(dim=1)[0].item()
        curve.append(probs[0, idx].item())

    if len(curve) < n_steps:
        curve = curve + [curve[-1]]*(n_steps-len(curve))
    curve = torch.tensor(curve[:n_steps]).unsqueeze(0)
    return curve

if __name__ == "__main__":
    main()
    print("done")
