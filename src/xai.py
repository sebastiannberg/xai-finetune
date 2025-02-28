import torch


def _run_forward(model, inputs: torch.Tensor, target: int):
    print("inside run forward func")
    output = model(inputs)
    print("after model(inputs)")
    # shape = (batch_size,) after slice out output[:, target]
    return output[:, target]

def _compute_gradients(model, inputs, target_idx):
    print("inside compute grads func before the with")
    with torch.autograd.set_grad_enabled(True):
        print("before run forward func")
        output = _run_forward(model, inputs, target_idx)
        print("before output.sum()")
        scalar_output = output.sum()
        print("after output.sum()")

        first_block = model.blocks[0]
        print("first_block", first_block)
        torch.autograd.grad(scalar_output, first_block.attn.attn)
        print("print after autograd.grad")

        for block in model.blocks:
            print("for block")
            grads = block.attn.attn.grad
            print(grads.size()) # (batch, head, seq, seq)
        print()

    return grads

def attribute(model: torch.nn.Module, data_loader_interpret: torch.utils.data.DataLoader, num_classes: int):
    """
    Loop over each class index, compute the attention gradients for every
    batch in data_loader_interpret, and accumulate/average them.
    """
    model.eval()
    print("model.eval")
    device = next(model.parameters()).device
    print("device")

    class_grads = []

    for class_idx in range(num_classes):
        accum_grad = None
        count_samples = 0

        print("class idx", class_idx)
        for fbank, _ in data_loader_interpret:
            print("in for fbank loop")
            fbank = fbank.to(device)
            print("after fbank to device")

            grads = _compute_gradients(model, fbank, class_idx)


#     print("Final Attention Gradient Shape:", total_attention_gradient.size(), "\n")
#     return total_attention_gradient  # Shape: (num_classes, blocks, heads, emb, emb)
