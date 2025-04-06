from tqdm import tqdm
from torch.cuda.amp import autocast

import models_vit as models_vit


def baseline_one_epoch(model, device, optimizer, criterion, scaler, data_loader_train, epoch, max_epoch):
    model.train()

    total_loss = 0.0
    for fbank, label in tqdm(data_loader_train, desc=f"Training [Epoch {epoch+1}/{max_epoch}]", leave=False, position=1):
        fbank = fbank.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(fbank)
            loss = criterion(logits, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(data_loader_train)

def ifi_one_epoch():
    pass

def et_one_epoch():
    pass





# import torch
# import torch.utils.data
# import torch.nn.functional as F
# from typing import List
# from tqdm import tqdm
# import os
# import matplotlib.pyplot as plt
# from pathlib import Path
# from datetime import datetime
#
# from utils import plot_attention_heatmap
#
#
# PROJECT_ROOT = Path(__file__).parent.parent.absolute()
# IMG_PATH = os.path.join(PROJECT_ROOT, 'img')
# os.makedirs(IMG_PATH, exist_ok=True)
#
# def _compute_gradients(model, inputs, class_idx):
#     model.zero_grad()
#     print(inputs.size())
#
#     with torch.autograd.set_grad_enabled(True):
#         # Forward
#         logits = model(inputs)  # shape: (batch_size, num_classes)
#
#         target_logits = logits[:, class_idx] # shape: (batch_size,)
#
#         scalar_output = target_logits.sum()
#
#         # Make sure to 'retain_grad()' for each attention block
#         for block in model.blocks:
#             if hasattr(block.attn, 'attn') and block.attn.attn is not None:
#                 if block.attn.attn.requires_grad:
#                     block.attn.attn.retain_grad()
#
#         # Backprop
#         scalar_output.backward()
#
#         # Collect attention grads
#         all_grads = []
#         first_sample_grads = []
#         for block in model.blocks:
#             if hasattr(block.attn, 'attn') and block.attn.attn.grad is not None:
#                 # first_sample_grad = block.attn.attn.grad[0].detach().clone()
#                 # block_idx = len(all_grads)
#                 # fig = plot_attention_heatmap(first_sample_grad, title=f"Block {block_idx} - Single Sample Grad")
#                 # fig.savefig(os.path.join(IMG_PATH, f"{int(time.time())}.png"))
#                 # plt.close(fig)
#                 # Sum across batch dimension
#                 all_grads.append(block.attn.attn.grad.detach().clone().sum(dim=0))
#
#                 first_sample_grads.append(block.attn.attn.grad[0].detach().clone())
#             else:
#                 print(f"Warning: No gradients for block {block}")
#                 # Add a placeholder of zeros with the same shape
#                 if len(all_grads) > 0:
#                     all_grads.append(torch.zeros_like(all_grads[0]))
#                 else:
#                     print("No valid gradients found in any block")
#                     return None
#
#         # Result shape: (num_blocks, num_heads, seq_len, seq_len)
#         stacked_grads = torch.stack(all_grads, dim=0)
#         stacked_first_sample = torch.stack(first_sample_grads, dim=0)
#         # print(stacked_first_sample.size()) # (12, 12, 257, 257)
#         orig_img = inputs[0]
#         orig_img = orig_img.detach().cpu().squeeze(0).transpose(0, 1).numpy()
#         timestamp = datetime.now().strftime('%m%d%H%M')
#
#         fig_orig, ax_orig = plt.subplots(figsize=(20, 5))
#         ax_orig.imshow(orig_img, cmap='gray', origin='lower', aspect='auto')
#         ax_orig.axis('off')
#         plt.tight_layout()
#
#         filename_orig = os.path.join(IMG_PATH, f"orig_{timestamp}.png")
#         fig_orig.savefig(filename_orig)
#         plt.close(fig_orig)
#
#         for block_idx in range(stacked_first_sample.shape[0]):
#             attn_map = stacked_first_sample[block_idx].mean(dim=0) # shape: (seq_len, seq_len)
#             attn_tensor = attn_map.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
#             attn_resized = F.interpolate(attn_tensor, size=(128, 512), mode='bilinear', align_corners=False)
#             attn_resized = attn_resized.squeeze().cpu().numpy()
#             attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
#
#             fig, ax = plt.subplots(figsize=(20, 5))
#             ax.imshow(orig_img, cmap='gray', origin='lower', aspect='auto')
#             ax.imshow(attn_norm, cmap='hot', alpha=0.2)
#             ax.axis('off')
#             plt.tight_layout()
#             filename = os.path.join(IMG_PATH, f"block{block_idx}_grad_heatmap_{timestamp}.png")
#             plt.savefig(filename)
#             plt.close(fig)
#
#             fig = plot_attention_heatmap(attn_map, title=f"Block {block_idx} - Single Sample Grad")
#             filename = os.path.join(IMG_PATH, f"block{block_idx}_{timestamp}.png")
#             fig.savefig(filename)
#             plt.close(fig)
#
#         raise ValueError("stop")
#
#         # Remove retained gradients to clean up
#         for block in model.blocks:
#             if hasattr(block.attn, 'attn') and block.attn.attn is not None:
#                 if block.attn.attn.grad is not None:
#                     block.attn.attn.grad = None
#
#         model.zero_grad()
#
#     return stacked_grads
#
# def attribute(model: torch.nn.Module, class_loaders: List[torch.utils.data.DataLoader]):
#     # Freeze dropout and batch norm with model.eval()
#     model.eval()
#
#     device = next(model.parameters()).device
#
#     class_grads = []
#     for class_idx, loader in tqdm(enumerate(class_loaders), desc="Attention Gradients", leave=False, total=len(class_loaders)):
#         accum_grads = None
#         total_samples = 0
#
#         for fbank, label in loader:
#             fbank = fbank.to(device)
#             label = label.to(device)
#
#             # Validation
#             labels_int = label.argmax(dim=1)
#             unique_labels = labels_int.unique()
#             assert unique_labels.numel() == 1 and unique_labels.item() == class_idx, (
#                 f"Loader mismatch! Found labels {unique_labels.tolist()} but expected single class {class_idx}"
#             )
#
#             # Compute grads
#             grads_sum = _compute_gradients(model, fbank, class_idx)
#
#             fig = plot_attention_heatmap(grads_sum, title="grads_sum attention heatmap")
#             fig.savefig(os.path.join(IMG_PATH, f"{int(time.time())}.png"))
#             plt.close(fig)
#
#             if accum_grads is not None:
#                 accum_grads += grads_sum
#             else:
#                 accum_grads = grads_sum
#             total_samples += fbank.size(0)
#
#         class_grad = accum_grads / float(total_samples)
#         class_grads.append(class_grad)
#
#     # Return a tensor of (class, block, head, seq, seq)
#     return torch.stack(class_grads, dim=0)
