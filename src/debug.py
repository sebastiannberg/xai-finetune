import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.models.layers import to_2tuple
import logging
import random
import os
from pathlib import Path
import matplotlib.pyplot as plt

from dataset_urban import UrbanDataset
import models_vit as models_vit


# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
CKPT_PATH = os.path.join(PROJECT_ROOT, 'ckpt')
LOGS_PATH = os.path.join(PROJECT_ROOT, 'logs')
URBAN_PATH = os.path.join(PROJECT_ROOT, 'data', 'UrbanSound8K')

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_PATH, 'finetune.log'), mode='a')
    ]
)
logger = logging.getLogger()

# Plot functions
def forward_histogram(tensor, name, pre_softmax=False, post_softmax=False):
    print(f'histogram for {name}')
    if tensor.dim() == 4:
        b, h, s1, s2 = tensor.shape
        # Flatten the last two dimensions -> (batch, head, s1*s2)
        tensor_flat = tensor.view(b, h, -1)
        for head in range(h):
            # For a given head, flatten over batch and flattened seq dims
            head_data = tensor_flat[:, head, :].flatten().cpu().detach().numpy()
            plt.figure(figsize=(10, 6))
            plt.hist(head_data, bins=100, alpha=0.7)
            if pre_softmax:
                title_str = f"Pre-Softmax {name} - Head {head}"
                head_file = f"pre_softmax_{name}_head{head}.png"
            elif post_softmax:
                title_str = f"Post-Softmax {name} - Head {head}"
                head_file = f"post_softmax_{name}_head{head}.png"
            else:
                title_str = f"{name} - Head {head}"
                head_file = f"{name}_head{head}.png"
            plt.title(title_str)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(PROJECT_ROOT, 'img', 'forward', head_file))
            plt.close()

def backward_histograms(grads: torch.Tensor):
    num_blocks = grads.size(0)
    num_heads = grads.size(1)
    seq_len = grads.size(2)
    logger.info(f"blocks: {num_blocks}, heads: {num_heads}, seq: {seq_len}")

    save_dir = os.path.join(PROJECT_ROOT, 'img', 'backward')
    os.makedirs(save_dir, exist_ok=True)

    for block_idx in range(num_blocks):
        block_grad = grads[block_idx]  
        for head_idx in range(num_heads):
            print(f'histogram for backward {block_idx}, {head_idx}')
            head_grad = block_grad[head_idx, :, :]
            head_grad = head_grad.cpu().numpy().flatten()

            plt.figure(figsize=(10, 6))
            plt.hist(head_grad, bins=100, alpha=0.7)
            title_str = f"Grad Block {block_idx}, Head {head_idx}"
            plt.title(title_str)
            plt.xlabel("Gradient Value")
            plt.ylabel("Frequency")
            plt.tight_layout()

            file_name = f"block_{block_idx}_head_{head_idx}.png"
            plt.savefig(os.path.join(save_dir, file_name))
            plt.close()

def plot_avg_token_attention(tensor, name):
    print(f'plot average token attention for {name}')
    attn_np = tensor.detach().cpu().numpy()[0]
    num_heads, S, _ = attn_np.shape

    plt.figure(figsize=(10, 6))
    for head in range(num_heads):
        avg_token_attn = attn_np[head].mean(axis=0)  # shape: (S,)
        plt.plot(np.arange(S), avg_token_attn, label=f"Head {head}")

    title_str = f"Avg Token Attention {name}"
    file_name = f"avg_token_attention_{name}.png"
    plt.xlabel("Token Index (Key)")
    plt.ylabel("Average Attention Weight")
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'img', 'forward', file_name))
    plt.close()

def plot_avg_attn_grad(grads):
    save_dir = os.path.join(PROJECT_ROOT, 'img', 'backward')
    os.makedirs(save_dir, exist_ok=True)

    avg_block_head = grads.mean(dim=(0, 1))
    query_avg = avg_block_head.mean(dim=0)
    print("plot average attention gradient values")

    plt.figure(figsize=(10, 6))
    x_vals = np.arange(query_avg.size(0))
    plt.plot(x_vals, query_avg.cpu().numpy(), label="Avg Grad across blocks, heads, queries")
    plt.xlabel("Key Token Index")
    plt.ylabel("Average Gradient")
    plt.title("Average Attention Gradient (summarizing all blocks/heads)")
    plt.legend()
    plt.tight_layout()

    file_name = "avg_attn_grad.png"
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()

class PatchEmbed_new(nn.Module):

    def __init__(self, img_size, patch_size, in_chans, embed_dim, stride):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

        _, _, h, w = self.get_output_shape(img_size)
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

def get_args():
    parser = argparse.ArgumentParser(description='Finetune with XAI on UrbanSound8K dataset')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of target classes')
    parser.add_argument('--target_length', type=int, default=512, help='Number of time frames for fbank')
    parser.add_argument('--checkpoint', type=str, default='epoch_1_20250305_233315.pth', help='Filename for model checkpoint to load before fine-tuning')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    parser.add_argument('--seed', type=int, default=0, help='To control the random seed for reproducibility')
    parser.add_argument('--debug_mode', type=str, default='forward', help='"forward" for forward pass inspection, "backward" for gradient inspection')
    return parser.parse_args()

def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fix the seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Optimal settings for speed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Reproducibility inside workers
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    dataset_debug = UrbanDataset(
        root=URBAN_PATH,
        fold=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        mixup_prob=0.5,
        roll_mag_aug=True,
        target_length=args.target_length,
        freqm=48,
        timem=192,
        num_classes=args.num_classes
    )
    data_loader_debug = DataLoader(
        dataset_debug,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    )

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
    checkpoint = torch.load(os.path.join(CKPT_PATH, args.checkpoint), map_location='cpu')
    checkpoint_state_dict = checkpoint['model_state_dict']
    with torch.no_grad():
        model.patch_embed.proj.weight.copy_(checkpoint_state_dict['patch_embed.proj.weight'])
        model.patch_embed.proj.bias.copy_(checkpoint_state_dict['patch_embed.proj.bias'])
        model.pos_embed.data.copy_(checkpoint_state_dict['pos_embed'])

    msg = model.load_state_dict(checkpoint_state_dict, strict=False)
    logger.info(msg)

    model.to(device)
    logger.info(f'Device: {device}')

    module_names = {id(mod): name for name, mod in model.named_modules()}

    def forward_hook(module, input, output):
        in_tensor = input[0] if isinstance(input, tuple) else input
        full_name = module_names.get(id(module), module._get_name())
        module_type = module.__class__.__name__
        logger.info(
            f"Layer [{full_name}] ({module_type}):\n"
            f"    Input shape: {in_tensor.shape}\n"
            f"    Output shape: {output.shape if hasattr(output, 'shape') else type(output)}\n"
            f"    Max: {output.max().item():.4f}\n"
            f"    Min: {output.min().item():.4f}\n"
            f"    Mean: {output.mean().item():.4f}\n"
            f"    Std: {output.std().item():.4f}"
        )

        if hasattr(module, 'attn_presoftmax'):
            pre = module.attn_presoftmax
            logger.info(
                f"Attention Pre-Softmax"
                f"    Shape: {pre.shape}\n"
                f"    Max: {pre.max().item():.4f}\n"
                f"    Min: {pre.min().item():.4f}\n"
                f"    Mean: {pre.mean().item():.4f}\n"
                f"    Std: {pre.std().item():.4f}"
            )
            forward_histogram(pre, full_name, pre_softmax=True)
        if hasattr(module, 'attn_postsoftmax'):
            post = module.attn_postsoftmax
            logger.info(
                f"Attention Post-Softmax"
                f"    Shape: {post.shape}\n"
                f"    Max: {post.max().item():.4f}\n"
                f"    Min: {post.min().item():.4f}\n"
                f"    Mean: {post.mean().item():.4f}\n"
                f"    Std: {post.std().item():.4f}"
            )
            forward_histogram(post, full_name, post_softmax=True)
            plot_avg_token_attention(post, full_name)

    def backward_hook(module, grad_input, grad_output):
        full_name = module_names.get(id(module), module._get_name())
        module_type = module.__class__.__name__

        if grad_input and grad_input[0] is not None:
            logger.info(
                f"Backward Hook [{full_name}] ({module_type}):\n"
                f"    Grad Input shape: {grad_input[0].shape}\n"
                f"    Grad Input Max: {grad_input[0].max().item()}\n"
                f"    Grad Input Min: {grad_input[0].min().item()}\n"
                f"    Grad Input Mean: {grad_input[0].mean().item()}\n"
                f"    Grad Input Std: {grad_input[0].std().item()}"
            )
        else:
            logger.info(
                f"Backward Hook [{full_name}] ({module_type}): grad_input is None "
                f"or grad_input[0] is None."
            )

    for _, module in model.named_modules():
        if args.debug_mode == 'forward':
            module.register_forward_hook(forward_hook)
        elif args.debug_mode == 'backward':
            module.register_full_backward_hook(backward_hook)

    model.train()
    fbank, label = next(iter(data_loader_debug))
    fbank = fbank.to(device)
    label = label.to(device)
    logger.info(f'fbank.size() {fbank.size()}')
    logger.info(f'label.size() {label.size()}')
    if args.debug_mode == 'forward':
        _ = model(fbank)
    elif args.debug_mode == 'backward':
        model.zero_grad()
        logits = model(fbank)
        class_idx = label.argmax()
        target_logit = logits[:, class_idx]

        # Make sure to 'retain_grad()' for each attention block
        for block in model.blocks:
            if hasattr(block.attn, 'attn') and block.attn.attn is not None:
                if block.attn.attn.requires_grad:
                    block.attn.attn.retain_grad()

        target_logit.backward()

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

        grads = torch.stack(all_grads, dim=0)
        grads = grads.squeeze(1)
        # Result shape: (num_blocks, num_heads, seq_len, seq_len)
        backward_histograms(grads)
        # TODO: plot of average received attention for grads, but only one plot, that means sum or average over blocks and heads
        plot_avg_attn_grad(grads)

        # Log model parameters gradients stats
        for name, param in reversed(list(model.named_parameters())):
            if param.grad is not None:
                logger.info(
                    f"Param {name}\n"
                    f"Grad shape: {param.grad.shape}\n"
                    f"Grad max: {param.grad.max()}\n"
                    f"Grad min: {param.grad.min()}\n"
                    f"Grad mean: {param.grad.mean()}\n"
                    f"Grad std: {param.grad.std()}"
                )

if __name__ == '__main__':
    main()
    print('done')
