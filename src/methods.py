import torch
from torch import Tensor
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
from pathlib import Path

import models_vit as models_vit


def baseline_one_epoch(manager, epoch):
    manager.model.train()

    total_loss = 0.0
    for fbank, label, filepath in tqdm(manager.data_loader_train, desc=f"Training [Epoch {epoch+1}/{manager.args.epochs}]", leave=False, position=1):
        fbank = fbank.to(manager.device)
        label = label.to(manager.device)

        manager.optimizer.zero_grad()

        with autocast():
            logits, attention = manager.model(fbank, return_attention=True)
            loss = manager.criterion(logits, label)

        manager.scaler.scale(loss).backward()
        manager.scaler.step(manager.optimizer)
        manager.scaler.update()

        total_loss += loss.item()

        for idx, item in enumerate(filepath):
            base_name = Path(item).name
            if base_name in manager.watched_filenames:
                manager.epoch_metrics["avg_received_attn_cls"][epoch].append(attention[idx].detach().cpu().numpy().mean(axis=(0, 1, 2))[0])
                if epoch == 0:
                    pure_fbank, _, _ = manager.data_loader_train.dataset.get_item_by_filename(base_name)
                    manager.plotter.plot_spectrogram(pure_fbank[0].detach().cpu().squeeze(0).numpy().T, base_name)
                if epoch + 1 in manager.plot_epochs:
                    manager.plotter.plot_attention_heatmap(attention[idx].detach().cpu().numpy(), base_name, epoch)
                    manager.plotter.plot_avg_received_attention(attention[idx].detach().cpu().numpy(), base_name, epoch)

    return total_loss / len(manager.data_loader_train)

def ifi_one_epoch(manager, epoch):
    manager.model.train()
    if epoch < manager.args.start_epoch - 1:
        # Training without interpretability loss
        manager.logger.info(f"Training without interpretability loss at epoch {epoch+1}")
        train_loss = baseline_one_epoch(manager, epoch)
    else:
        # Training with interpretability loss
        total_loss = 0.0
        entropies = []
        manager.logger.info(f"Training with interpretability loss at epoch {epoch+1}")
        for fbank, label, filepath in tqdm(manager.data_loader_train, desc=f"Training [Epoch {epoch+1}/{manager.args.epochs}]", leave=False, position=1):
            fbank = fbank.to(manager.device)
            label = label.to(manager.device)

            manager.optimizer.zero_grad()

            with autocast():
                logits, attention = manager.model(fbank, return_attention=True)
                classification_loss = manager.criterion(logits, label)

                label_indices = torch.argmax(label, dim=1)
                selected_attention_grads = manager.class_attention_grads[label_indices, ...]

                # Process gradients or target
                pre_attention_interpret = attention * selected_attention_grads
                if manager.args.grad_processing_mode == "temp_scaling":
                    post_attention_interpret = pre_attention_interpret / manager.args.temperature
                elif manager.args.grad_processing_mode == "relu":
                    post_attention_interpret = torch.relu(pre_attention_interpret)
                elif manager.args.grad_processing_mode == "standardize":
                    mean = pre_attention_interpret.mean(dim=-1, keepdim=True)
                    std = pre_attention_interpret.std(dim=-1, keepdim=True, unbiased=False) + 1e-9
                    post_attention_interpret = (pre_attention_interpret - mean) / std
                elif manager.args.grad_processing_mode == "percentile_clamping":
                    threshold = torch.quantile(pre_attention_interpret.abs(), q=manager.args.percentile, dim=-1, keepdim=True)
                    post_attention_interpret = torch.where(pre_attention_interpret.abs() < threshold, torch.zeros_like(pre_attention_interpret), pre_attention_interpret)
                elif manager.args.grad_processing_mode == "l1_amplification":
                    post_attention_interpret = pre_attention_interpret + manager.args.l1_lambda * pre_attention_interpret.abs()
                post_attention_interpret = post_attention_interpret.softmax(dim=-1)

                # Logger warning if nearly uniform
                softmax_mean = post_attention_interpret.mean(dim=-1)
                softmax_std = post_attention_interpret.std(dim=-1, unbiased=False)
                global_mean = softmax_mean.mean().item()
                global_std = softmax_std.mean().item()
                global_min = post_attention_interpret.min().item()
                global_max = post_attention_interpret.max().item()
                if global_std < 1e-7:
                    manager.logger.warning(
                        f"attention_interpret (target) appears nearly uniform "
                        f"(mean={global_mean:.5f}, std={global_std:.5e}, "
                        f"min={global_min:.5e}, max={global_max:.5e})"
                    )

                # Select blocks
                if manager.args.which_blocks == "all":
                    attention_sel = attention
                    attention_interpret_sel = post_attention_interpret
                elif manager.args.which_blocks == "first":
                    attention_sel = attention[:, :1, ...]
                    attention_interpret_sel = post_attention_interpret[:, :1, ...]
                elif manager.args.which_blocks == "last":
                    attention_sel = attention[:, -1:, ...]
                    attention_interpret_sel = post_attention_interpret[:, -1:, ...]

                # Activate or deactivate cls token in interpretability loss
                if manager.args.cls_deactivated:
                    attention_sel = attention_sel.clone()
                    attention_interpret_sel = attention_interpret_sel.clone()
                    attention_sel[..., 0, :] = 0
                    attention_sel[..., :, 0] = 0
                    attention_interpret_sel[..., 0, :] = 0
                    attention_interpret_sel[..., :, 0] = 0

                entropy = -(attention_interpret_sel.detach().clamp(min=1e-12) * attention_interpret_sel.detach().clamp(min=1e-12).log()).sum(dim=-1).mean().item()
                entropies.append(entropy)

                # Compute interpretability loss (cross entropy) over selected blocks
                interpret_loss = -(attention_interpret_sel.detach() * (attention_sel + 1e-12).log()).sum(dim=-1).mean()
                # Total loss
                loss = (manager.args.alpha * classification_loss) + ((1 - manager.args.alpha) * interpret_loss)

            manager.scaler.scale(loss).backward()
            manager.scaler.step(manager.optimizer)
            manager.scaler.update()

            total_loss += loss.item()

            for idx, item in enumerate(filepath):
                base_name = Path(item).name
                if base_name in manager.watched_filenames:
                    manager.epoch_metrics["avg_received_attn_cls"][epoch].append(attention[idx].detach().cpu().numpy().mean(axis=(0, 1, 2))[0])
                    if epoch + 1 in manager.plot_epochs:
                        manager.plotter.plot_attention_heatmap(attention[idx].detach().cpu().numpy(), base_name, epoch)
                        manager.plotter.plot_attention_heatmap(post_attention_interpret[idx].detach().cpu().numpy(), base_name, epoch, mode="attention_interpret")
                        manager.plotter.plot_avg_received_attention(attention[idx].detach().cpu().numpy(), base_name, epoch)
                        manager.plotter.plot_avg_received_attention(post_attention_interpret[idx].detach().cpu().numpy(), base_name, epoch, mode="attention_interpret")

        # Log the entropy for targets
        manager.logger.info(f"Interpretability target entropy: {sum(entropies)/len(entropies):.5f}")

        train_loss = total_loss / len(manager.data_loader_train)

    # Calculate attention gradients
    class_attention_grads = attribute(manager, epoch)
    manager.class_attention_grads = class_attention_grads

    return train_loss

def compute_attention_rollout(raw_attention: Tensor, mode: str) -> Tensor:
    """
    Given the raw per-head attention weights from a Transformer,
    returns the attention‐rollout map of shape [B, S, S] as in
    Abnar & Zuidema (2020).

    raw_attention[b, i, h] is the (S×S) attention from head h at layer i.
    We:
      1) avg over heads to get one [B, S, S] per layer
      2) add the identity (0.5*I + 0.5*A) at every layer
      3) recursively matmul shallow->deep

    Parameters
    ----------
    raw_attention : Tensor
        The raw attention weights of shape [batch_size, num_layers, num_heads, seq_length, seq_length].
    mode : str
        * "original"      — merge heads first then rollout with 0.5·I + 0.5·A at each layer;
        * "head_rollout"  — rollout each head separately using 0.5·I + 0.5·A at each layer, then stack per-head maps.

    Returns
    -------
    Tensor
        If mode == "original": a Tensor of shape [batch_size, seq_length, seq_length] containing the merged-head rollout map.
        If mode == "head_rollout": a Tensor of shape [batch_size, num_heads, seq_length, seq_length] containing one rollout map per head.
    """
    _, num_layers, num_heads, seq_length, _ = raw_attention.shape
    identity = torch.eye(seq_length, device=raw_attention.device).unsqueeze(0) # [1, S, S]

    if mode == "original":
        # 1) Merge heads by averaging: one [B, S, S] per layer
        merged_per_layer = []
        for layer_idx in range(num_layers):
            # raw_attention[:, layer_idx] is [B, H, S, S]
            avg_attention = raw_attention[:, layer_idx].mean(dim=1) # [B, S, S]
            merged_per_layer.append(avg_attention)

        # 2) Rollout: start at layer 0 without residual
        rollout_map = merged_per_layer[0]

        # 3) For layers 1…L-1, do 0.5A + 0.5I then matmul
        for layer_attention in merged_per_layer[1:]:
            residual_attention = 0.5 * layer_attention + 0.5 * identity # [B, S, S]
            rollout_map = torch.matmul(residual_attention, rollout_map) # [B, S, S]

        return rollout_map

    elif mode == "head_rollout":
        # 1) For each head, build and rollout its own A_res = 0.5A + 0.5I
        per_head_rollouts = []
        for head_idx in range(num_heads):
            # build per-layer residual-augmented matrices
            per_layer_attention = []
            for layer_idx in range(num_layers):
                single_head_attention = raw_attention[:, layer_idx, head_idx]  # [B, S, S]
                attention_with_residual = 0.5 * single_head_attention + 0.5 * identity
                per_layer_attention.append(attention_with_residual)

            # 2) shallow->deep product for this head
            head_rollout = per_layer_attention[0]
            for att_res in per_layer_attention[1:]:
                head_rollout = torch.matmul(att_res, head_rollout)

            per_head_rollouts.append(head_rollout)

        # 3) stack into [batch_size, num_heads, seq_length, seq_length]
        return torch.stack(per_head_rollouts, dim=1)

    else:
        raise ValueError(f"Unsupported mode {mode}")

def k_sigma_thresholding(head_rollouts, sigma_k):
    head_mean = head_rollouts.mean(dim=(-1, -2)) # -> (batch, head)
    head_std  = head_rollouts.std(dim=(-1, -2), unbiased=False) # -> (batch, head)
    threshold = head_mean + sigma_k * head_std # -> (batch, head)
    binary_maps = (head_rollouts >= threshold.unsqueeze(-1).unsqueeze(-1)).float() # -> (batch, head, seq, seq)
    return binary_maps, head_mean.unsqueeze(-1).unsqueeze(-1)

def et_one_epoch(manager, epoch):
    manager.model.train()
    if epoch < manager.args.start_epoch - 1:
        train_loss = baseline_one_epoch(manager, epoch)
    else:
        total_loss = 0.0
        zero_fracs = []
        dynamic_ranges = []
        for fbank, label, filepath in tqdm(manager.data_loader_train, desc=f"Training [Epoch {epoch+1}/{manager.args.epochs}]", leave=False, position=1):
            fbank = fbank.to(manager.device)
            label = label.to(manager.device)

            with autocast():
                _, attention = manager.model(fbank, return_attention=True)
            head_rollouts = compute_attention_rollout(attention, mode="head_rollout") # (batch, head, seq, seq)
            binary_maps, map_weights = k_sigma_thresholding(head_rollouts, manager.args.sigma_k)

            # Compute zero fractions statistics
            zero_frac = (binary_maps == 0).float().mean(dim=(-1, -2)) # -> (batch, head)
            zero_frac_mean = zero_frac.mean(dim=(0, 1))
            zero_fracs.append(zero_frac_mean.item()) 

            weighted_sum = (map_weights * binary_maps).sum(dim=1) # -> (batch, seq, seq)

            mins = weighted_sum.amin(dim=(-1, -2), keepdim=True)
            maxs = weighted_sum.amax(dim=(-1, -2), keepdim=True)
            dynamic_ranges.append((maxs - mins).mean().item())
            normalized_maps = (weighted_sum - mins) / (maxs - mins + 1e-10) # -> (batch, seq, seq)

            patch_scores = normalized_maps[:, 0, 1:] # -> (batch, seq-1) only cls row, and drop cls column

            h_patches, w_patches = manager.model.patch_embed.patch_hw
            patch_grid = patch_scores.view(-1, 1, h_patches, w_patches) # -> (batch, 1, h_patches, w_patches)

            h_img, w_img = manager.model.patch_embed.img_size
            saliency_maps = torch.nn.functional.interpolate(patch_grid, size=(h_img, w_img), mode="bilinear") # -> (batch, 1, h_img, w_img)

            saliency_mins = saliency_maps.amin(dim=(-1, -2), keepdim=True)
            saliency_maxs = saliency_maps.amax(dim=(-1, -2), keepdim=True)
            normalized_saliency_maps = (saliency_maps - saliency_mins) / (saliency_maxs - saliency_mins + 1e-10)

            # Gradually introduce explanations
            progress = (epoch - manager.args.start_epoch) / (manager.args.epochs - manager.args.start_epoch)
            alpha_saliency = min(max(progress, 0.0), 1.0)
            augmented_input = fbank * (1 - alpha_saliency) + (fbank * normalized_saliency_maps.detach()) * alpha_saliency

            manager.optimizer.zero_grad()
            with autocast():
                logits, attention = manager.model(augmented_input, return_attention=True)
                loss = manager.criterion(logits, label)

            manager.scaler.scale(loss).backward()
            manager.scaler.step(manager.optimizer)
            manager.scaler.update()

            total_loss += loss.item()

            # Plots
            for idx, item in enumerate(filepath):
                base_name = Path(item).name
                if base_name in manager.watched_filenames:
                    manager.epoch_metrics["avg_received_attn_cls"][epoch].append(attention[idx].detach().cpu().numpy().mean(axis=(0, 1, 2))[0])
                    if epoch == 0:
                        pure_fbank, _, _ = manager.data_loader_train.dataset.get_item_by_filename(base_name)
                        manager.plotter.plot_spectrogram(pure_fbank[0].detach().cpu().squeeze(0).numpy().T, base_name)
                    if epoch + 1 in manager.plot_epochs:
                        manager.plotter.plot_saliency_map(normalized_saliency_maps[idx][0].detach().cpu().numpy().T, base_name, epoch)
                        pure_fbank, _, _ = manager.data_loader_train.dataset.get_item_by_filename(base_name)
                        manager.plotter.plot_heatmap(pure_fbank[0].detach().cpu().squeeze(0).numpy().T, normalized_saliency_maps[idx][0].detach().cpu().numpy().T, base_name, epoch)
                        manager.plotter.plot_augmented_input(augmented_input[idx][0].detach().cpu().numpy().T, base_name, epoch)

        # Log statistics
        manager.logger.info(f"Average zero fraction in binary maps: {np.mean(zero_fracs):.4f}")
        manager.logger.info(f"Average one fraction in binary maps: {1 - np.mean(zero_fracs):.4f}")
        manager.logger.info(f"Average dynamic range in binary weighted sums: {np.mean(dynamic_ranges):.4f}")

        train_loss = total_loss / len(manager.data_loader_train)

    return train_loss

def compute_gradients(manager, inputs, class_idx, filepath, epoch):
    manager.model.zero_grad()

    with torch.autograd.set_grad_enabled(True):
        # Forward
        logits = manager.model(inputs)  # shape: (batch_size, num_classes)
        target_logits = logits[:, class_idx] # shape: (batch_size,)
        scalar_output = target_logits.sum()

        # Ensure 'retain_grad()' for each attention block before backprop
        for block in manager.model.blocks:
            if hasattr(block.attn, 'attn') and block.attn.attn is not None:
                if block.attn.attn.requires_grad:
                    block.attn.attn.retain_grad()
                    block.attn.attn_pre_softmax.retain_grad()

        # Backprop
        scalar_output.backward()

        # Collect attention grads and plot
        all_grads = []
        tmp_snr = {Path(item).name: [] for item in filepath}
        tmp_snr_pre = {Path(item).name: [] for item in filepath}
        for i, block in enumerate(manager.model.blocks):
            if hasattr(block.attn, 'attn') and block.attn.attn.grad is not None:
                # Plotting
                for idx, item in enumerate(filepath):
                    base_name = Path(item).name
                    if base_name in manager.watched_filenames or base_name in ["noise"]:
                        # Calculate the SNR value of the attention gradient
                        grads_abs = block.attn.attn.grad[idx].detach().clone().cpu().abs()
                        mu  = grads_abs.mean()
                        std = grads_abs.std(unbiased=False) + 1e-12
                        snr_block = (mu / std).item()
                        tmp_snr[base_name].append(snr_block)

                        grads_attn_pre_softmax = block.attn.attn_pre_softmax.grad[idx].detach().clone().cpu().abs()
                        mu_pre = grads_attn_pre_softmax.mean()
                        std_pre = grads_attn_pre_softmax.std(unbiased=False) + 1e-12
                        snr_pre_block = (mu_pre / std_pre).item()
                        tmp_snr_pre[base_name].append(snr_pre_block)

                        should_plot = False
                        if isinstance(epoch, int) and (epoch + 1) in manager.plot_epochs:
                            should_plot = True
                        if epoch in ["noise_test"]:
                            should_plot = True
                        if should_plot:
                            manager.plotter.plot_attention_gradient(block.attn.attn.grad[idx].detach().clone().cpu().numpy(), base_name, epoch, i)
                # Sum and store in list for later return
                all_grads.append(block.attn.attn.grad.detach().clone().sum(dim=0)) # Gradients are summed here along batch dim
            else:
                manager.logger.warning(f"Warning: No gradients for block {block}")
                # Add a placeholder of zeros with the same shape
                if len(all_grads) > 0:
                    all_grads.append(torch.zeros_like(all_grads[0]))
                else:
                    manager.logger.error("No valid gradients found in any block")
                    return None

        stacked_grads = torch.stack(all_grads, dim=0) # shape: (num_blocks, num_heads, seq_len, seq_len)

        # Remove retained gradients to clean up
        for block in manager.model.blocks:
            if hasattr(block.attn, 'attn') and block.attn.attn is not None:
                if block.attn.attn.grad is not None:
                    block.attn.attn.grad = None
        manager.model.zero_grad()

        # Calculate SNR for epoch
        for filename, snr_list in tmp_snr.items():
            if snr_list:
                epoch_snr = torch.tensor(snr_list).mean().item()
                manager.snr_values[filename].append(epoch_snr)
        for filename, snr_pre_list in tmp_snr_pre.items():
            if snr_pre_list:
                epoch_snr_pre = torch.tensor(snr_pre_list).mean().item()
                manager.snr_pre_values[filename].append(epoch_snr_pre)

    return stacked_grads

def attribute(manager, epoch):
    # Freeze dropout and batch norm with model.eval()
    manager.model.eval()

    class_grads = []
    for class_idx, loader in tqdm(enumerate(manager.class_loaders), desc="Attention Gradients", leave=False, total=len(manager.class_loaders)):
        accum_grads = None
        total_samples = 0

        for fbank, label, filepath in loader:
            fbank = fbank.to(manager.device)
            label = label.to(manager.device)

            # Ensure same label for all samples
            labels_int = label.argmax(dim=1)
            unique_labels = labels_int.unique()
            assert unique_labels.numel() == 1 and unique_labels.item() == class_idx, (
                f"Loader mismatch! Found labels {unique_labels.tolist()} but expected single class {class_idx}"
            )

            grads_sum = compute_gradients(manager, fbank, class_idx, filepath, epoch)

            if accum_grads is not None:
                accum_grads += grads_sum
            else:
                accum_grads = grads_sum
            total_samples += fbank.size(0)

        class_grad = accum_grads / float(total_samples)
        class_grads.append(class_grad)

    return torch.stack(class_grads, dim=0) # shape: (class, block, head, seq, seq)
