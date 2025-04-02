import torch
import torch.utils.data
from typing import List
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from pathlib import Path
import time

from utils import plot_attention_heatmap


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
IMG_PATH = os.path.join(PROJECT_ROOT, 'img')
os.makedirs(IMG_PATH, exist_ok=True)

def _compute_gradients(model, inputs, class_idx):
    model.zero_grad()

    with torch.autograd.set_grad_enabled(True):
        # Forward
        logits = model(inputs)  # shape: (batch_size, num_classes)

        target_logits = logits[:, class_idx] # shape: (batch_size,)

        scalar_output = target_logits.sum()

        # Make sure to 'retain_grad()' for each attention block
        for block in model.blocks:
            if hasattr(block.attn, 'attn') and block.attn.attn is not None:
                if block.attn.attn.requires_grad:
                    block.attn.attn.retain_grad()

        # Backprop
        scalar_output.backward()

        # Collect attention grads
        all_grads = []
        first_sample_grads = []
        for block in model.blocks:
            if hasattr(block.attn, 'attn') and block.attn.attn.grad is not None:
                first_sample_grad = block.attn.attn.grad[0].detach().clone()
                # block_idx = len(all_grads)
                # fig = plot_attention_heatmap(first_sample_grad, title=f"Block {block_idx} - Single Sample Grad")
                # fig.savefig(os.path.join(IMG_PATH, f"{int(time.time())}.png"))
                # plt.close(fig)
                # Sum across batch dimension
                all_grads.append(block.attn.attn.grad.detach().clone().sum(dim=0))

                first_sample_grads.append(block.attn.attn.grad[0].detach().clone())
            else:
                print(f"Warning: No gradients for block {block}")
                # Add a placeholder of zeros with the same shape
                if len(all_grads) > 0:
                    all_grads.append(torch.zeros_like(all_grads[0]))
                else:
                    print("No valid gradients found in any block")
                    return None

        # Result shape: (num_blocks, num_heads, seq_len, seq_len)
        stacked_grads = torch.stack(all_grads, dim=0)
        stacked_first_sample = torch.stack(first_sample_grads, dim=0)
        print(stacked_first_sample.size())

        # Remove retained gradients to clean up
        for block in model.blocks:
            if hasattr(block.attn, 'attn') and block.attn.attn is not None:
                if block.attn.attn.grad is not None:
                    block.attn.attn.grad = None

        model.zero_grad()

    return stacked_grads

def attribute(model: torch.nn.Module, class_loaders: List[torch.utils.data.DataLoader]):
    # Freeze dropout and batch norm with model.eval()
    model.eval()

    device = next(model.parameters()).device

    class_grads = []
    for class_idx, loader in tqdm(enumerate(class_loaders), desc="Attention Gradients", leave=False, total=len(class_loaders)):
        accum_grads = None
        total_samples = 0

        for fbank, label in loader:
            fbank = fbank.to(device)
            label = label.to(device)

            # Validation
            labels_int = label.argmax(dim=1)
            unique_labels = labels_int.unique()
            assert unique_labels.numel() == 1 and unique_labels.item() == class_idx, (
                f"Loader mismatch! Found labels {unique_labels.tolist()} but expected single class {class_idx}"
            )

            # Compute grads
            grads_sum = _compute_gradients(model, fbank, class_idx)

            fig = plot_attention_heatmap(grads_sum, title="grads_sum attention heatmap")
            fig.savefig(os.path.join(IMG_PATH, f"{int(time.time())}.png"))
            plt.close(fig)

            if accum_grads is not None:
                accum_grads += grads_sum
            else:
                accum_grads = grads_sum
            total_samples += fbank.size(0)

        class_grad = accum_grads / float(total_samples)
        class_grads.append(class_grad)

    # Return a tensor of (class, block, head, seq, seq)
    return torch.stack(class_grads, dim=0)
