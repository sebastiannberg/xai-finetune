import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from timm.models.layers import to_2tuple, trunc_normal_
import logging
import random
import os
from pathlib import Path

from dataset_urban import UrbanDataset
import models_vit as models_vit
from grad import attribute

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
    parser.add_argument('--checkpoint', type=str, default='pretrained.pth', help='Filename for model checkpoint to load before fine-tuning')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    parser.add_argument('--seed', type=int, default=0, help='To control the random seed for reproducibility')
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
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    # Check if positional embeddings need to be interpolated
    if 'pos_embed' in checkpoint_model:
        if checkpoint_model['pos_embed'].shape != model.pos_embed.shape:
            logger.info(f"Interpolating positional embeddings from {checkpoint_model['pos_embed'].shape} to {torch.Size([1, 257, 768])}")
            pos_embed_checkpoint = checkpoint_model['pos_embed']  # [1, old_num_tokens, embed_dim]
            cls_token = pos_embed_checkpoint[:, :1, :]            # [1, 1, embed_dim]
            pos_tokens = pos_embed_checkpoint[:, 1:, :]           # [1, old_num_tokens-1, embed_dim]

            # Determine the original grid shape
            num_tokens_pretrained = pos_tokens.shape[1]
            if num_tokens_pretrained == 512:
                grid_shape_pretrained = (8, 64)  # Known grid shape from pretraining
            else:
                grid_size = int(np.sqrt(num_tokens_pretrained))
                grid_shape_pretrained = (grid_size, grid_size)

            # Reshape from (1, num_tokens, embed_dim) -> (1, grid_height, grid_width, embed_dim)
            pos_tokens = pos_tokens.reshape(1, grid_shape_pretrained[0], grid_shape_pretrained[1], -1)
            # Permute to (1, embed_dim, grid_height, grid_width) for interpolation
            pos_tokens = pos_tokens.permute(0, 3, 1, 2)

            # New grid size from your custom patch embedding (e.g., (32, 8) in your case)
            new_grid_size = model.patch_embed.patch_hw

            # Interpolate using bilinear interpolation
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=new_grid_size,
                mode='bilinear',
                align_corners=False
            )
            # Permute back and reshape to (1, new_num_tokens, embed_dim)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, -1, pos_embed_checkpoint.shape[-1])

            # Concatenate the class token back
            new_pos_embed = torch.cat((cls_token, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

    # Remove the classification head weights if they don't match the current model's output shape
    # This prevents shape mismatch issues when fine-tuning on a different number of classes
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            logger.info(f"Removing key {k} from pretrained checkpoint due to shape mismatch.")
            del checkpoint_model[k]
    # Load the remaining pre-trained weights into the model
    # strict=False allows partial loading (ignores missing keys like the removed head)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)
    # Reinitialize the classification head with small random values
    # This ensures the new classification layer starts learning from scratch
    trunc_normal_(model.head.weight, std=2e-5)
    model.to(device)

    logger.info(f'Device: {device}')

    # Debug
    model.train()
    fbank, label = next(iter(data_loader_debug))
    fbank = fbank.to(device)
    label = label.to(device)
    logger.info(f'fbank.size() {fbank.size()}')
    logger.info(f'label.size() {label.size()}')

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

    for _, module in model.named_modules():
        module.register_forward_hook(forward_hook)

    logits = model(fbank)

if __name__ == '__main__':
    main()
    print('done')

