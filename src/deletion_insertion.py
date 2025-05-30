import argparse
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
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
    "baseline_with_augmentations_lr10-5_wd10-5_bs8": "/home/sebastian/dev/xai-finetune/results/final_experiments/baseline_with_augmentations_lr10-5_wd10-5_bs8/ckpt/epoch_60.pth"
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

def compute_deletion(inputs, saliency, model, label, n_steps=20):
    flat_sal = saliency.view(-1).detach().cpu().numpy()

    # 95, 90, …, 0   (same order as the official implementation)
    percents   = np.linspace(100, 0, n_steps, endpoint=False)   # e.g. 100,95,…,5
    thresholds = np.percentile(flat_sal, percents)

    cls   = label.argmax(dim=1)[0].item()

    # step 0: original confidence
    with torch.no_grad():
        orig_conf = F.softmax(model(inputs), dim=1)[0, cls]
        blank_conf = F.softmax(model(torch.zeros_like(inputs)), dim=1)[0, cls]

    curve = [orig_conf.item()]          # start at original

    for thresh in thresholds:
        # mask out the TOP-saliency pixels above the current threshold
        mask_flat = (saliency.view(-1) > thresh).to(inputs.device).float()
        x_flat    = inputs.view(1, inputs.shape[1], -1) * mask_flat
        x_mask    = x_flat.view_as(inputs)

        prob = F.softmax(model(x_mask), dim=1)[0, cls]
        curve.append(prob.item())

    curve.append(blank_conf.item())     # final step = fully blank
    return torch.tensor(curve).unsqueeze(0)

def compute_insertion(inputs, saliency, model, label, n_steps=20):
    flat_sal = saliency.view(-1).detach().cpu().numpy()
    percents = np.linspace(100, 0, n_steps, endpoint=False)     # same grid as deletion
    thresholds = np.percentile(flat_sal, percents)

    cls   = label.argmax(dim=1)[0].item()
    blank = torch.zeros_like(inputs)

    # endpoints
    with torch.no_grad():
        blank_conf = F.softmax(model(blank),   dim=1)[0, cls]
        orig_conf  = F.softmax(model(inputs),  dim=1)[0, cls]

    curve = [blank_conf.item()]                     # start at blank

    for thresh in thresholds:
        mask_flat = (saliency.view(-1) > thresh).to(inputs.device).float()
        inputs_flat = inputs.view(1, inputs.shape[1], -1)
        blank_flat  = blank.view(1, inputs.shape[1], -1)
        x_ins_flat  = blank_flat * (1 - mask_flat) + inputs_flat * mask_flat
        x_ins       = x_ins_flat.view_as(inputs)

        prob = F.softmax(model(x_ins), dim=1)[0, cls]
        curve.append(prob.item())

    curve.append(orig_conf.item())                  # final = original
    return torch.tensor(curve).unsqueeze(0)

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

        # single‐batch smoke test
        fbank, label, _ = next(iter(data_loader_val))
        fbank = fbank.to(device); label = label.to(device)

        # get the two curves
        saliency = attention_rollout(model, model(fbank, return_attention=True)[1])
        del_curve = compute_deletion(fbank, saliency, model, label, n_steps=100)  # (1, n_steps)
        ins_curve = compute_insertion(fbank, saliency, model, label, n_steps=100)  # (1, n_steps)

        # compute the “ground truth” endpoints
        with torch.no_grad():
            orig_logits  = model(fbank)
            blank       = torch.zeros_like(fbank)
            blank_logits = model(blank)
            class_index        = label.argmax(dim=1)[0].item()            # get the integer class
            orig_conf  = torch.softmax(orig_logits,  dim=1)[0, class_index]
            blank_conf = torch.softmax(blank_logits, dim=1)[0, class_index]

        # check deletion curve
        print("del_curve:", del_curve[0, 0].item(), "…", del_curve[0, -1].item())
        print(" expected orig_conf=", orig_conf.item(), 
              " blank_conf=",   blank_conf.item())

        # check insertion curve
        print("ins_curve:", ins_curve[0, 0].item(), "…", ins_curve[0, -1].item())
        print(" expected blank_conf=",   blank_conf.item(), 
              " orig_conf=", orig_conf.item())

        rand_sal = torch.rand_like(saliency)
        rand_del = compute_deletion(fbank, rand_sal, model, label, n_steps=100)
        rand_ins = compute_insertion(fbank, rand_sal, model, label, n_steps=100)
        print(f"RANDOM   deletion mean={rand_del.mean():.4f}  insertion mean={rand_ins.mean():.4f}")

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

        def curve_auc(curve_tensor):
            c = curve_tensor.squeeze(0).cpu().numpy()
            return float(np.trapz(c, dx=1.0) / (len(c) - 1))

        del_auc = curve_auc(del_r)        # .= concatenated tensor (n_samples, n_steps+2)
        ins_auc = curve_auc(ins_r)
        gra_del_auc = curve_auc(del_g)
        gra_ins_auc = curve_auc(ins_g)

        # move everything to NumPy
        del_r_np = del_r.detach().cpu().numpy()
        ins_r_np = ins_r.detach().cpu().numpy()
        del_g_np = del_g.detach().cpu().numpy()
        ins_g_np = ins_g.detach().cpu().numpy()

        # compute “official” deletion/insertion scores:
        #   mean over steps for each sample, then mean over samples
        del_rollout_score = float(del_r_np.mean(axis=1).mean())
        ins_rollout_score = float(ins_r_np.mean(axis=1).mean())
        del_grad_score    = float(del_g_np.mean(axis=1).mean())
        ins_grad_score    = float(ins_g_np.mean(axis=1).mean())

        results[name] = {
            "deletion_rollout_score": del_rollout_score,
            "insertion_rollout_score": ins_rollout_score,
            "deletion_gradients_score": del_grad_score,
            "insertion_gradients_score": ins_grad_score,
            "del_auc":          del_auc,
            "ins_auc":          ins_auc,
            "grad_del_auc":     gra_del_auc,
            "grad_ins_auc":     gra_ins_auc
        }

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
