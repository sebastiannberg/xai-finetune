import argparse
import torch
import torch.nn as nn
import torchaudio
from timm.models.layers import to_2tuple
import os
from pathlib import Path

import models_vit as models_vit


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
    parser.add_argument('--checkpoint', type=str, default='epoch_40_20250218_222750.pth', help='Filename for model to load for inference')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of target classes')
    parser.add_argument('--target_length', type=int, default=512, help='Number of time frames for fbank')
    parser.add_argument('--files', nargs='+', required=True, help='List of audio file paths for inference')
    return parser.parse_args()

def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    print(f'Device: {device}\n')

    model.eval()

    idx_to_class = {
        0: 'air_conditioner',
        1: 'car_horn',
        2: 'children_playing',
        3: 'dog_bark',
        4: 'drilling',
        5: 'engine_idling',
        6: 'gun_shot',
        7: 'jackhammer',
        8: 'siren',
        9: 'street_music'
    }

    def compute_fbank(audio_file):
        waveform, sr = torchaudio.load(audio_file)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=sr,
            use_energy=False,
            htk_compat=True,
            window_type='hanning',
            num_mel_bins=128,
            frame_shift=10,
            dither=0.0
        )

        # Normalize
        fbank = (fbank - -3.85) / 3.85

        # Pad or truncate fbank to fixed length
        p = args.target_length - fbank.shape[0]
        if p > 0:
            # Padding
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            # Cutting
            fbank = fbank[0:args.target_length, :]

        return fbank.unsqueeze(0)

    for audio_file in args.files:
        fbank = compute_fbank(audio_file)
        fbank = fbank.unsqueeze(0)
        fbank = fbank.to(device)

        with torch.no_grad():
            logits = model(fbank)
            probs = torch.softmax(logits, dim=-1)
            print('Probs: ', probs)

        pred_class_id = torch.argmax(probs, dim=1).item()
        pred_probability = probs[0, pred_class_id].item()
        pred_label = idx_to_class[pred_class_id]

        print(f"File: {audio_file}")
        print(f"  Predicted Class: {pred_label} (ID: {pred_class_id})")
        print(f"  Probability:     {pred_probability:.4f}\n")

if __name__ == '__main__':
    main()

