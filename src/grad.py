import torch


def _run_forward(model, inputs: torch.Tensor, target: int):
    output = model(inputs)
    # shape = (batch_size,) after slice out output[:, target]
    return output[:, target]

def _compute_gradients(model, inputs, target_idx):
    model.zero_grad()

    with torch.autograd.set_grad_enabled(True):
        output = _run_forward(model, inputs, target_idx)

        for block in model.blocks:
            if hasattr(block.attn, 'attn') and block.attn.attn is not None:
                if block.attn.attn.requires_grad:
                    block.attn.attn.retain_grad()

        scalar_output = output.sum()

        scalar_output.backward()

        all_grads = []
        for block in model.blocks:
            if hasattr(block.attn, 'attn') and block.attn.attn.grad is not None:
                all_grads.append(block.attn.attn.grad.detach().clone())
            else:
                print(f"Warning: No gradients for block {block}")
                # Add a placeholder of zeros with the same shape
                if len(all_grads) > 0:
                    all_grads.append(torch.zeros_like(all_grads[0]))
                else:
                    print("No valid gradients found in any block")
                    return None

        # Result shape: (batch_size, num_blocks, num_heads, seq_len, seq_len)
        stacked_grads = torch.stack(all_grads, dim=1)
        print(stacked_grads.size())

        model.zero_grad()

    return stacked_grads

def attribute(model: torch.nn.Module, data_loader_interpret: torch.utils.data.DataLoader, num_classes: int):
    """
    Loop over each class index, compute the attention gradients for every
    batch in data_loader_interpret, and accumulate/average them.
    """
    # Freeze dropout and batch norm with model.eval()
    model.eval()

    device = next(model.parameters()).device

    class_grads = []

    for class_idx in range(num_classes):
        accum_grad = None
        count_samples = 0

        for fbank, _ in data_loader_interpret:
            fbank = fbank.to(device)

            grads = _compute_gradients(model, fbank, class_idx)

    return grads

