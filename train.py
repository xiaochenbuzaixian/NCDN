import os
import math
import yaml
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from data.shin_dataset import get_dataloaders
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion

def cosine_lr(epoch, warmup, total, peak_lr, min_lr=1e-5):
    if epoch < warmup:
        return min_lr + (peak_lr - min_lr) * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def main():
    with open("configs/shin_baseline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    modality = cfg["data"]["target_modality"]
    processed_dir = cfg["data"]["processed_dir"]
    use_hrf = cfg["data"].get("eeg_hrf", True)

    topo_dict = torch.load(os.path.join(processed_dir, f"topo_{modality.lower()}.pt"), map_location="cpu")
    topo_ef, topo_fe = topo_dict["topo_ef"].to(device), topo_dict["topo_fe"].to(device)

    train_loader, val_loader = get_dataloaders(
        processed_dir, modality,
        batch_size=cfg["data"]["batch_size"],
        train_ratio=cfg["data"]["train_ratio"]
    )

    unet = SCDMUnet(time_dim=128, base_channels=64, n_blocks=6).to(device)
    model = SCDMDiffusion(unet, timesteps=cfg["model"]["timesteps"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)
    epochs = cfg["train"]["epochs"]
    warmup = cfg["train"].get("warmup_epochs", 20)
    peak_lr = cfg["train"]["lr"]
    best_score = -999.0

    print(f">>> 启动 SCDM (U-Net + SCG + MTR + 神经血管先验) 训练 | 设备: {device}")

    for epoch in range(1, epochs + 1):
        lr = cosine_lr(epoch - 1, warmup, epochs, peak_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        sums = {"total": 0.0, "noise_mse": 0.0, "prior_mse": 0.0}
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:03d}/{epochs:03d}]", leave=False)

        for batch in pbar:
            e_in = batch["eeg_hrf" if use_hrf else "eeg"].to(device)
            f_0 = batch["fnirs"].to(device)
            optimizer.zero_grad()
            loss, l_dict = model.compute_loss(e_in, f_0, topo_ef, topo_fe)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            for k in sums:
                sums[k] += l_dict[k]
            pbar.set_postfix({"n_mse": f"{l_dict['noise_mse']:.4f}", "p_mse": f"{l_dict['prior_mse']:.4f}"})

        n = len(train_loader)
        print(f"Ep [{epoch:03d}/{epochs:03d}] | LR: {lr:.2e} | Noise_MSE: {sums['noise_mse']/n:.4f} | Prior_MSE: {sums['prior_mse']/n:.4f}")

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            reals, synths = [], []
            with torch.no_grad():
                for vb in val_loader:
                    ve = vb["eeg_hrf" if use_hrf else "eeg"].to(device)
                    vf = vb["fnirs"].numpy()
                    sf = model.sample(ve, topo_ef, topo_fe, steps=20).cpu().numpy()
                    reals.append(vf)
                    synths.append(sf)
            reals = np.concatenate(reals, axis=0)
            synths = np.concatenate(synths, axis=0)
            val_mse = float(np.mean((reals - synths) ** 2))
            corrs = [float(((r - r.mean()) * (s - s.mean())).sum() / (np.sqrt(((r - r.mean())**2).sum() * ((s - s.mean())**2).sum()) + 1e-12)) for r, s in zip(reals, synths)]
            val_spcc = float(np.mean(corrs))
            val_cpcc, _ = pearsonr(reals.mean(axis=(0, 1)), synths.mean(axis=(0, 1)))
            score = val_spcc + 0.5 * val_cpcc - 0.5 * val_mse
            print(f"  --> [验证集 N={len(reals)}] MSE: {val_mse:.4f} | 单试次 PCC: {val_spcc:.4f} | 曲线 PCC: {val_cpcc:.4f}")

            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), os.path.join(cfg["train"]["checkpoint_dir"], f"scdm_{modality.lower()}_best.pt"))
                print(f"  ★ 已更新最佳模型 (Val MSE: {val_mse:.4f}, Curve PCC: {val_cpcc:.4f})")

        if epoch == epochs:
            torch.save(model.state_dict(), os.path.join(cfg["train"]["checkpoint_dir"], f"scdm_{modality.lower()}_ep{epoch}.pt"))

if __name__ == "__main__":
    main()