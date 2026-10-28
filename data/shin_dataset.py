import os
import torch
from torch.utils.data import Dataset, DataLoader

class ShinEEGfNIRSDataset(Dataset):
    def __init__(self, eeg, eeg_hrf, fnirs, labels):
        self.eeg = eeg
        self.eeg_hrf = eeg_hrf
        self.fnirs = fnirs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "eeg": self.eeg[idx],          # (30, 4000) 用于下游 EEGNet 分类
            "eeg_hrf": self.eeg_hrf[idx],  # (30, 256)  用于 SCDM 扩散模型条件引导
            "fnirs": self.fnirs[idx],      # (36, 256)
            "label": self.labels[idx]
        }

def get_dataloaders(processed_dir, modality="HbR", batch_size=16, train_ratio=0.8, seed=42):
    eeg = torch.load(os.path.join(processed_dir, "eeg_epochs.pt"))
    hrf_path = os.path.join(processed_dir, "eeg_hrf_epochs.pt")
    eeg_hrf = torch.load(hrf_path) if os.path.exists(hrf_path) else eeg
    fnirs = torch.load(os.path.join(processed_dir, f"fnirs_{modality.lower()}_epochs.pt"))
    labels = torch.load(os.path.join(processed_dir, "labels.pt"))
    N = len(labels)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    n_train = int(N * train_ratio)
    tr_idx, val_idx = perm[:n_train], perm[n_train:]
    train_ds = ShinEEGfNIRSDataset(eeg[tr_idx], eeg_hrf[tr_idx], fnirs[tr_idx], labels[tr_idx])
    val_ds = ShinEEGfNIRSDataset(eeg[val_idx], eeg_hrf[val_idx], fnirs[val_idx], labels[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader