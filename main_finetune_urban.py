import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from timm.models.layers import to_2tuple

from dataset_urban import UrbanDataset
import models_vit as models_vit


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
    parser = argparse.ArgumentParser(description='Finetune on UrbanSound8K dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of target classes')
    parser.add_argument('--target_length', type=int, default=512, help='Number of time frames for fbank')
    parser.add_argument('--checkpoint', type=str, default='/cluster/projects/uasc/sebastian/xai-finetune/ckpt/pretrained.pth', help='Path to the pretrained model checkpoint')
    return parser.parse_args()

def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fix the seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    dataset_train = UrbanDataset(
        root='/cluster/projects/uasc/sebastian/xai-finetune/data/UrbanSound8K',
        fold=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        mixup_prob=0.5,
        roll_mag_aug=True,
        target_length=args.target_length,
        freqm=48,
        timem=192,
        num_classes=args.num_classes
    )
    dataset_val = UrbanDataset(
        root='/cluster/projects/uasc/sebastian/xai-finetune/data/UrbanSound8K',
        fold=[10],
        mixup_prob=0.0,
        roll_mag_aug=False,
        target_length=args.target_length,
        freqm=0,
        timem=0,
        num_classes=args.num_classes
    )
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=8,
        num_workers=8,
        drop_last=True
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=8,
        num_workers=8,
        drop_last=False
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
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    # Remove the classification head weights if they don't match the current model's output shape
    # This prevents shape mismatch issues when fine-tuning on a different number of classes
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint due to shape mismatch.")
            del checkpoint_model[k]
    # Load the remaining pre-trained weights into the model
    # strict=False allows partial loading (ignores missing keys like the removed classifier head)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    # Reinitialize the classification head with small random values
    # This ensures the new classification layer starts learning from scratch
    trunc_normal_(model.head.weight, std=2e-5)
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = nn.BCEWithLogitsLoss()

    print(f'Start training for {args.epochs} epochs')
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        pass

    total_time = time.time() - start_time
    print(f'Total training time: {total_time / 60:.2f} minutes')

if __name__ == '__main__':
    main()

