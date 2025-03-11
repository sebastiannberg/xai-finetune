import torch
import torch.utils.data
from typing import List
from tqdm import tqdm


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

        scalar_output = target_logits.sum()

        # Backprop
        scalar_output.backward()

        # Collect attention grads
        all_grads = []
        for block in model.blocks:
            if hasattr(block.attn, 'attn') and block.attn.attn.grad is not None:
                # Sum across batch dimension
                all_grads.append(block.attn.attn.grad.detach().clone().sum(dim=0))
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

            if accum_grads is not None:
                accum_grads += grads_sum
            else:
                accum_grads = grads_sum
            total_samples += fbank.size(0)

        class_grad = accum_grads / float(total_samples)
        class_grads.append(class_grad)

    # Return a tensor of (class, block, head, emb, emb)
    return torch.stack(class_grads, dim=0)
