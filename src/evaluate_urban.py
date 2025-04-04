import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.models.layers import to_2tuple
from sklearn.metrics import accuracy_score, f1_score
import os
from pathlib import Path

from dataset_urban import UrbanDataset
import models_vit as models_vit
from grad import _compute_gradients


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
CKPT_PATH = os.path.join(PROJECT_ROOT, 'ckpt')
URBAN_PATH = os.path.join(PROJECT_ROOT, 'data', 'UrbanSound8K')

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
    parser.add_argument('--checkpoint', type=str, default='epoch_60_20250222_192308.pth', help='Filename for model to load for evaluation')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of target classes')
    parser.add_argument('--target_length', type=int, default=512, help='Number of time frames for fbank')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of worker threads for data loading')
    return parser.parse_args()

def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_val = UrbanDataset(
        root=URBAN_PATH,
        fold=[10],
        mixup_prob=0.0,
        roll_mag_aug=False,
        target_length=args.target_length,
        freqm=0,
        timem=0,
        num_classes=args.num_classes
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
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
    checkpoint = torch.load(os.path.join(CKPT_PATH, args.checkpoint), map_location='cpu')
    checkpoint_state_dict = checkpoint['model_state_dict']
    with torch.no_grad():
        model.patch_embed.proj.weight.copy_(checkpoint_state_dict['patch_embed.proj.weight'])
        model.patch_embed.proj.bias.copy_(checkpoint_state_dict['patch_embed.proj.bias'])
        model.pos_embed.data.copy_(checkpoint_state_dict['pos_embed'])

    msg = model.load_state_dict(checkpoint_state_dict, strict=False)
    print(msg)

    model.to(device)

    print(f'Device: {device}')

    model.eval()
    all_preds = []
    all_labels = []
    # with torch.no_grad():
    for fbank_val, label_val in data_loader_val:
        first_sample, first_label = fbank_val[0], label_val[0]
        # fbank_val = fbank_val.to(device)
        # label_val = label_val.to(device)

        sample = first_sample.to(device)
        sample = sample[None, ...]
        # sample.view(1, 1, -1, -1)
        # sample.unsqueeze(0).unsqueeze(0)
        label_int = first_label.argmax().item()
        _compute_gradients(model=model, inputs=sample, class_idx=label_int)

        # logits_val = model(fbank_val)

        # preds = torch.argmax(logits_val, dim=1)
        # true_classes = torch.argmax(label_val, dim=1)

        # all_preds.append(preds.cpu())
        # all_labels.append(true_classes.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    val_accuracy = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Val Accuracy:  {val_accuracy:.4f}")
    print(f"Val F1:        {val_f1:.4f}")

if __name__ == '__main__':
    main()

