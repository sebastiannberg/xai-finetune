from typing import List, Tuple, Callable
import torch


def _apply_grad_requirements(tensors: Tuple[torch.Tensor]) -> List[bool]:
    """
    Sets requires_grad to True where needed, and ensures
    all grads are set to zero. Returns a list of flags
    representing whether or not each tensor originally required grad.
    """
    original_requires_grad = []
    for t in tensors:
        original_requires_grad.append(t.requires_grad)
        t.requires_grad_(True)
        if t.grad is not None:
            t.grad.zero_()
    return original_requires_grad

def _undo_grad_requirements(tensors: Tuple[torch.Tensor], original_requires_grad: List[bool]) -> None:
    """
    Reverts the requires_grad flags to their original states
    and zeros out any gradients.
    """
    for t, orig_req_grad in zip(tensors, original_requires_grad):
        t.requires_grad_(orig_req_grad)
        if t.grad is not None:
            t.grad.zero_()

def _run_forward(forward_fn: Callable, inputs: torch.Tensor, target: int):
    """
    Utility that calls forward_fn with return_attention=True
    and returns (attention, output_for_that_target).
    """
    # forward_fn should return: attention, output
    # where attention is shape: (batch_size, block, head, seq_len, seq_len)
    # and output is shape: (batch_size, num_classes)
    output, attention = forward_fn(inputs, return_attention=True)
    # Pick the relevant logit (or probability) for target
    # shape = (batch_size,) if you slice out output[:, target]
    print("attention requires_grad:", attention.requires_grad)
    print("attention grad_fn:", attention.grad_fn)
    return attention, output[:, target]

def _compute_gradients(forward_fn: Callable, inputs, target_idx):
    with torch.autograd.set_grad_enabled(True):
        # attention = (batch_size, block, head, seq_len, seq_len)
        # output = (?, ?)
        attention, output = _run_forward(forward_fn, inputs, target_idx)
        print(attention.size())
        print(output.size())
        print()

        # Mark attention to require grad, so we can call torch.autograd.grad(...)
        original_flags = _apply_grad_requirements((attention))

        # If output is shape (batch_size,), you may want a scalar to differentiate w.r.t.
        # Often, we sum over the batch so grad(...) returns a single gradient tensor.
        # E.g., sum the output (or mean) so that the gradient is well-defined as a single tensor:
        scalar_output = output.sum()

        grads = torch.autograd.grad(scalar_output, attention, retain_graph=True, allow_unused=True)

        _undo_grad_requirements((attention), original_flags)

    return grads[0]

def attribute(model: torch.nn.Module, data_loader_interpret: torch.utils.data.DataLoader, num_classes: int):
    """
    Loop over each class index, compute the attention gradients for every
    batch in data_loader_interpret, and accumulate/average them.
    """
    model.eval()
    device = next(model.parameters()).device

    class_grads = []

    for class_idx in range(num_classes):
        accum_grad = None
        count_samples = 0

        for fbank, _ in data_loader_interpret:
            fbank = fbank.to(device)

            grads = _compute_gradients(model.forward, fbank, class_idx)
            print(grads)
            print('grads size: ', grads.size())


#     print("Final Attention Gradient Shape:", total_attention_gradient.size(), "\n")
#     return total_attention_gradient  # Shape: (num_classes, blocks, heads, emb, emb)
