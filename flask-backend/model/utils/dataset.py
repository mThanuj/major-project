import os
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

DATASET_PATH = "./data"


def download_dataset():
    return torchaudio.datasets.SPEECHCOMMANDS(root=DATASET_PATH, download=True)


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        target_length: int = 100,
        n_mfcc: int = 40,
        melkwargs: dict = dict(),
        augment: bool = False,
        test_mode: bool = False,
    ):
        self.dataset_path = dataset_path
        self.target_length = target_length
        self.n_mfcc = n_mfcc
        self.augment = augment
        self.test_mode = test_mode

        base = os.path.join(dataset_path, "SpeechCommands/speech_commands_v0.02")
        self.classes = sorted(
            [
                d
                for d in os.listdir(base)
                if os.path.isdir(os.path.join(base, d)) and not d.startswith("_")
            ]
        )
        self.label_to_index = {lbl: i for i, lbl in enumerate(self.classes)}
        self.index_to_label = {i: lbl for lbl, i in self.label_to_index.items()}

        self.file_paths = []
        for lbl in self.classes:
            p = os.path.join(base, lbl)
            for f in os.listdir(p):
                if f.endswith(".wav"):
                    self.file_paths.append((os.path.join(p, f), lbl))

        melkwargs = {
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": n_mfcc,
            "f_min": 50,
            "f_max": 8000,
        }

        # Build the MFCC transform once
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=melkwargs,
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path, lbl = self.file_paths[idx]
        waveform, _ = torchaudio.load(path)

        # Augmentation
        if self.augment and not self.test_mode:
            if torch.rand(1).item() > 0.5:
                shift = int(waveform.shape[1] * 0.1 * torch.rand(1).item())
                waveform = torch.roll(waveform, shifts=shift, dims=1)
            if torch.rand(1).item() > 0.5:
                noise = 0.005 * torch.rand(1).item() * torch.randn_like(waveform)
                waveform = waveform + noise

        # Extract MFCCs
        mfcc = self.mfcc_transform(waveform).squeeze(0)  # [n_mfcc, T]

        # Pad or crop time to target_length
        T = mfcc.shape[1]
        if T < self.target_length:
            mfcc = F.pad(mfcc, (0, self.target_length - T))
        elif T > self.target_length:
            start = (T - self.target_length) // 2
            mfcc = mfcc[:, start : start + self.target_length]

        # Per-sample normalization
        m, s = mfcc.mean(), mfcc.std() + 1e-8
        mfcc = (mfcc - m) / s

        # Add channel dimension → [1, n_mfcc, target_length]
        mfcc = mfcc.unsqueeze(0)
        label_idx = self.label_to_index[lbl]

        return (mfcc, label_idx, lbl) if self.test_mode else (mfcc, label_idx)
