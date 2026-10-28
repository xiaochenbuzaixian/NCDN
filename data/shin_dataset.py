import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

class ShinEEGfNIRSDataset(Dataset):
    def __init__(self, eeg, eeg_hrf, fnirs, labels, indices):
        self.eeg = eeg[indices]
        self.eeg_hrf = eeg_hrf[indices]
        self.fnirs = fnirs[indices]
        self.labels = labels[indices]
        self.indices = indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "eeg": self.eeg[idx],
            "eeg_hrf": self.eeg_hrf[idx],
            "fnirs": self.fnirs[idx],
            "label": self.labels[idx]
        }

def get_dataloaders(processed_dir, modality="HbR", batch_size=16, train_ratio=0.7, seed=42):
    eeg = torch.load(os.path.join(processed_dir, "eeg_epochs.pt"))
    hrf_path = os.path.join(processed_dir, "eeg_hrf_epochs.pt")
    eeg_hrf = torch.load(hrf_path) if os.path.exists(hrf_path) else eeg
    fnirs = torch.load(os.path.join(processed_dir, f"fnirs_{modality.lower()}_epochs.pt"))
    labels = torch.load(os.path.join(processed_dir, "labels.pt"))

    split_path = os.path.join(processed_dir, "split_indices.json")
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            sp = json.load(f)
        tr_idx = torch.tensor(sp["train_indices"], dtype=torch.long)
        val_idx = torch.tensor(sp["val_indices"], dtype=torch.long)
    else:
        N = len(labels)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=g)
        n_train = int(N * train_ratio)
        tr_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = ShinEEGfNIRSDataset(eeg, eeg_hrf, fnirs, labels, tr_idx)
    val_ds = ShinEEGfNIRSDataset(eeg, eeg_hrf, fnirs, labels, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader