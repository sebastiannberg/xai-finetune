import torch
from torch.cuda.amp import autocast
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
                if epoch == 0:
                    pure_fbank, _, _ = manager.data_loader_train.dataset.get_item_by_filename(base_name)
                    manager.plotter.plot_spectrogram(pure_fbank[0].detach().cpu().squeeze(0).numpy().T, base_name)
                if epoch + 1 in manager.plot_epochs:
                    manager.plotter.plot_attention_heatmap(attention[idx].detach().cpu().numpy(), base_name, epoch)

    return total_loss / len(manager.data_loader_train)

def ifi_one_epoch(manager, epoch):
    manager.model.train()
    if epoch < 1:
        # Training without interpretability loss
        train_loss = baseline_one_epoch(manager, epoch)
    else:
        # Training with interpretability loss
        total_loss = 0.0
        for fbank, label, filepath in tqdm(manager.data_loader_train, desc=f"Training [Epoch {epoch+1}/{manager.args.epochs}]", leave=False, position=1):
            fbank = fbank.to(manager.device)
            label = label.to(manager.device)

            manager.optimizer.zero_grad()

            with autocast():
                logits, attention = manager.model(fbank, return_attention=True)
                classification_loss = manager.criterion(logits, label)

                label_indices = torch.argmax(label, dim=1)
                selected_attention_grads = manager.class_attention_grads[label_indices, ...]
                selected_attention_grads = manager.args.grad_scale * selected_attention_grads

                # attention_wo_cls = attention[:, :, :, 1:, 1:]
                # attention_wo_cls_softmaxed = attention_wo_cls.softmax(dim=-1)
                # grads_wo_cls = selected_attention_grads[:, :, :, 1:, 1:]

                pre_attention_interpret = attention * selected_attention_grads
                post_attention_interpret = pre_attention_interpret.softmax(dim=-1)

                # Cross Entropy
                interpret_loss = -(post_attention_interpret.detach() * (attention + 1e-12).log()).sum(dim=-1).mean()

                loss = (manager.args.alpha * classification_loss) + ((1 - manager.args.alpha) * interpret_loss)
                # loss = classification_loss

            manager.scaler.scale(loss).backward()
            manager.scaler.step(manager.optimizer)
            manager.scaler.update()

            total_loss += loss.item()

            for idx, item in enumerate(filepath):
                base_name = Path(item).name
                if base_name in manager.watched_filenames:
                    if epoch + 1 in manager.plot_epochs:
                        manager.plotter.plot_attention_heatmap(attention[idx].detach().cpu().numpy(), base_name, epoch)
                        manager.plotter.plot_attention_heatmap(post_attention_interpret[idx].detach().cpu().numpy(), base_name, epoch, mode="attention_interpret")

        train_loss = total_loss / len(manager.data_loader_train)

    # Calculate attention gradients
    class_attention_grads = attribute(manager)
    manager.class_attention_grads = class_attention_grads

    return train_loss

def et_one_epoch():
    pass

def compute_gradients(manager, inputs, class_idx):
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

        # Backprop
        scalar_output.backward()

        # Collect attention grads
        all_grads = []
        for block in manager.model.blocks:
            if hasattr(block.attn, 'attn') and block.attn.attn.grad is not None:
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

    return stacked_grads

def attribute(manager):
    # Freeze dropout and batch norm with model.eval()
    manager.model.eval()

    class_grads = []
    for class_idx, loader in tqdm(enumerate(manager.class_loaders), desc="Attention Gradients", leave=False, total=len(manager.class_loaders)):
        accum_grads = None
        total_samples = 0

        for fbank, label, _ in loader:
            fbank = fbank.to(manager.device)
            label = label.to(manager.device)

            # Ensure same label for all samples
            labels_int = label.argmax(dim=1)
            unique_labels = labels_int.unique()
            assert unique_labels.numel() == 1 and unique_labels.item() == class_idx, (
                f"Loader mismatch! Found labels {unique_labels.tolist()} but expected single class {class_idx}"
            )

            grads_sum = compute_gradients(manager, fbank, class_idx)

            if accum_grads is not None:
                accum_grads += grads_sum
            else:
                accum_grads = grads_sum
            total_samples += fbank.size(0)

        class_grad = accum_grads / float(total_samples)
        class_grads.append(class_grad)

    return torch.stack(class_grads, dim=0) # shape: (class, block, head, seq, seq)
