import os
import pandas as pd
import torchaudio
import torch
from torch.utils.data import Dataset
import random
import numpy as np


class UrbanDataset(Dataset):

    def __init__(self, root: str, fold: list, mixup_prob:float, roll_mag_aug: bool, target_length: int, freqm: int, timem: int, num_classes: int):
        self.root = root
        self.fold = fold
        self.mixup_prob = mixup_prob
        self.roll_mag_aug = roll_mag_aug
        self.target_length = target_length
        self.freqm = freqm
        self.timem = timem
        self.num_classes = num_classes

        self.metadata = pd.read_csv(os.path.join(root, 'metadata', 'UrbanSound8K.csv'))

        # Filter data based on selected folds
        self.metadata = self.metadata[self.metadata['fold'].isin(fold)]
        self.file_paths = self.metadata.apply(lambda row: os.path.join(root, 'audio', f"fold{row['fold']}", row['slice_file_name']), axis=1).tolist()
        self.labels = self.metadata['classID'].tolist()

        self.mean = -3.85
        self.std = 3.85

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        if random.random() < self.mixup_prob:
            # Apply mixup with probability 'self.mixup_prob'
            mix_idx = random.randint(0, len(self.file_paths) - 1)
            mix_path = self.file_paths[mix_idx]
            mix_label = self.labels[mix_idx]
            fbank, label = self._wav_to_fbank(file_path, label, mix_path, mix_label)
        else:
            # No mixup, just process a single file
            fbank, label = self._wav_to_fbank(file_path, label, None, None)

        return fbank, label

    def _roll_mag_aug(self, waveform):
        """
        Applies roll-and-magnitude augmentation to the waveform.

        The rolling simulates temporal shifts, and the scaling adjusts amplitude variation.

        Returns:
            torch.Tensor: Augmented waveform tensor.
        """
        waveform_np = waveform.numpy()
        shift_amount = np.random.randint(len(waveform_np))
        rolled_waveform = np.roll(waveform_np, shift_amount)
        magnitude_factor = np.random.beta(10, 10) + 0.5
        return torch.tensor(rolled_waveform * magnitude_factor)

    def _wav_to_fbank(self, file_path, label, mixup_path, mixup_label):
        if mixup_path is None:
            waveform, sr = torchaudio.load(file_path)
            waveform = waveform - waveform.mean()

            if self.roll_mag_aug:
                waveform = self._roll_mag_aug(waveform)

            # Convert label to one-hot vector
            label_one_hot = np.zeros(self.num_classes)
            label_one_hot[label] = 1.0
        else:
            waveform, sr = torchaudio.load(file_path)
            mixup_waveform, _ = torchaudio.load(mixup_path)

            waveform = waveform - waveform.mean()
            mixup_waveform = mixup_waveform - mixup_waveform.mean()

            if self.roll_mag_aug:
                waveform = self._roll_mag_aug(waveform)
                mixup_waveform = self._roll_mag_aug(mixup_waveform)

            # Ensure same length for mixup
            if waveform.shape[1] != mixup_waveform.shape[1]:
                if waveform.shape[1] > mixup_waveform.shape[1]:
                    # Padding
                    temp_wav = torch.zeros(1, waveform.shape[1])
                    temp_wav[0, 0:mixup_waveform.shape[1]] = mixup_waveform
                    mixup_waveform = temp_wav
                else:
                    # Cutting
                    mixup_waveform = mixup_waveform[:, :waveform.shape[1]]

            # Perform mixup
            mix_lambda = np.random.beta(10, 10)
            mix_waveform = mix_lambda * waveform + (1 - mix_lambda) * mixup_waveform
            waveform = mix_waveform - mix_waveform.mean()

            # Mixup label handling
            label_one_hot = np.zeros(self.num_classes)
            label_one_hot[label] += mix_lambda
            label_one_hot[mixup_label] += (1.0 - mix_lambda)
 
        # Convert waveform to log-mel spectrogram (fbank)
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

        # Normalize fbank first
        fbank = (fbank - self.mean) / self.std

        # Pad or truncate fbank to fixed length
        p = self.target_length - fbank.shape[0]
        if p > 0:
            # Padding
            m = torch.nn.zeropad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            # Cutting
            fbank = fbank[0:self.target_length, :]

        # Apply SpecAugment (freqm and timem)
        if self.freqm != 0:
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            fbank = freqm(fbank)
        if self.timem != 0:
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = timem(fbank)

        label_one_hot = torch.FloatTensor(label_one_hot)
        return fbank, label_one_hot

