import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from timm.models.layers import to_2tuple, trunc_normal_
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import logging
import sys
import os
import datetime
import random

from dataset_urban import UrbanDataset
import models_vit as models_vit


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('/cluster/projects/uasc/sebastian/xai-finetune/logs/finetune.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
print = logging.info

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
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
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
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

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

    # Check if positional embeddings need to be interpolated
    if 'pos_embed' in checkpoint_model:
        if checkpoint_model['pos_embed'].shape != model.pos_embed.shape:
            print("Interpolating positional embeddings from", checkpoint_model['pos_embed'].shape,
                  "to", model.pos_embed.shape)
            pos_embed_checkpoint = checkpoint_model['pos_embed']  # [1, old_num_tokens, embed_dim]
            cls_token = pos_embed_checkpoint[:, :1, :]            # [1, 1, embed_dim]
            pos_tokens = pos_embed_checkpoint[:, 1:, :]             # [1, old_num_tokens-1, embed_dim]

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
            print(f"Removing key {k} from pretrained checkpoint due to shape mismatch.")
            del checkpoint_model[k]
    # Load the remaining pre-trained weights into the model
    # strict=False allows partial loading (ignores missing keys like the removed head)
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

    start_time = time.time()
    best_val_loss = float('inf')
    best_model_path = None
    for epoch in range(args.epochs):
        model.train()

        total_train_loss = 0.0

        train_pbar = tqdm(data_loader_train, desc=f"Epoch [{epoch+1}/{args.epochs}] (Training)", leave=False)
        for fbank, label in train_pbar:
            fbank = fbank.to(device)
            label = label.to(device)

            logits = model(fbank)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(data_loader_train.dataset)

        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for fbank_val, label_val in data_loader_val:
                fbank_val = fbank_val.to(device)
                label_val = label_val.to(device)

                logits_val = model(fbank_val)
                loss_val = criterion(logits_val, label_val)
                total_val_loss += loss_val.item()

                preds = torch.argmax(logits_val, dim=1)
                true_classes = torch.argmax(label_val, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(true_classes.cpu())

        avg_val_loss = total_val_loss / len(data_loader_val.dataset)

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Epoch [{epoch+1}/{args.epochs}]")
        print(f"  Train Loss:    {avg_train_loss:.4f}")
        print(f"  Val Loss:      {avg_val_loss:.4f}")
        print(f"  Val Accuracy:  {val_accuracy:.4f}")
        print(f"  Val F1:        {val_f1:.4f}")
        print("-"*40)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_best_model_path = f'/cluster/projects/uasc/sebastian/xai-finetune/ckpt/best_model_{timestamp}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss
            }, new_best_model_path)
            print(f'New best model saved: {new_best_model_path}')

            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"Previous best model removed: {best_model_path}")
            best_model_path = new_best_model_path

    total_time = time.time() - start_time
    print(f'Total training time: {total_time / 60:.2f} minutes')

if __name__ == '__main__':
    main()

