import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class ShinEEGfNIRSDataset(Dataset):
    def __init__(self, processed_dir="datasets/processed", modality="HbR"):
        super().__init__()
        self.eeg = torch.load(os.path.join(processed_dir, "eeg_epochs.pt"))
        fnirs_file = "fnirs_hbr_epochs.pt" if modality == "HbR" else "fnirs_hbo_epochs.pt"
        self.fnirs = torch.load(os.path.join(processed_dir, fnirs_file))
        self.labels = torch.load(os.path.join(processed_dir, "labels.pt"))
        self.sub_ids = torch.load(os.path.join(processed_dir, "subject_ids.pt"))

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return {
            "eeg": self.eeg[idx],          # [30, 4000]
            "fnirs": self.fnirs[idx],      # [36, 256]
            "label": self.labels[idx],
            "sub_id": self.sub_ids[idx]
        }

def get_dataloaders(processed_dir="datasets/processed", modality="HbR", batch_size=16, train_ratio=0.8):
    dataset = ShinEEGfNIRSDataset(processed_dir, modality)
    train_len = int(len(dataset) * train_ratio)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader