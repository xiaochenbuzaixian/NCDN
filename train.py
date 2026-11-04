import os
import csv
import math
import time
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
    exp_id = f"SCDM_{modality}_{time.strftime('%Y%m%d_%H%M%S')}"

    topo_dict = torch.load(os.path.join(processed_dir, f"topo_{modality.lower()}.pt"), map_location="cpu")
    topo_ef, topo_fe = topo_dict["topo_ef"].to(device), topo_dict["topo_fe"].to(device)
    corr_ef = topo_dict.get("corr_ef", None)
    spatial_affinity = topo_dict.get("spatial_affinity", None)
    if corr_ef is not None:
        corr_ef = corr_ef.to(device)
    if spatial_affinity is not None:
        spatial_affinity = spatial_affinity.to(device)

    train_loader, val_loader = get_dataloaders(
        processed_dir, modality,
        batch_size=cfg["data"]["batch_size"],
        train_ratio=cfg["data"]["train_ratio"]
    )

    unet = SCDMUnet(
        time_dim=cfg["model"].get("time_dim", 64),
        base_channels=cfg["model"].get("base_channels", 32),
        n_blocks=6
    ).to(device)
    model = SCDMDiffusion(unet, timesteps=cfg["model"]["timesteps"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)
    epochs = cfg["train"]["epochs"]
    warmup = cfg["train"].get("warmup_epochs", 15)
    peak_lr = cfg["train"]["lr"]
    best_ddim_score = -999.0

    csv_path = os.path.join(cfg["train"]["output_dir"], f"train_log_{modality.lower()}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["exp_id", "epoch", "lr", "train_total", "train_noise_mse", "train_x0_mse", "val_ddim_mse", "val_spcc", "val_cpcc"])

    print(f">>> 启动 6-Block SCDM 训练 [{exp_id}] | 设备: {device} | 训练: {len(train_loader.dataset)} | 验证: {len(val_loader.dataset)}")

    for epoch in range(1, epochs + 1):
        lr = cosine_lr(epoch - 1, warmup, epochs, peak_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        sums = {"total": 0.0, "noise_mse": 0.0, "x0_mse": 0.0}
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:03d}/{epochs:03d}]", leave=False)

        for batch in pbar:
            e_in = batch["eeg_hrf" if use_hrf else "eeg"].to(device)
            f_0 = batch["fnirs"].to(device)
            optimizer.zero_grad()
            loss, l_dict = model.compute_loss(e_in, f_0, topo_ef, topo_fe, corr_ef=corr_ef, spatial_affinity=spatial_affinity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            for k in sums:
                sums[k] += l_dict[k]
            pbar.set_postfix({"n_mse": f"{l_dict['noise_mse']:.4f}", "x0_mse": f"{l_dict['x0_mse']:.4f}"})

        n = len(train_loader)
        tr_tot, tr_nmse, tr_x0 = sums["total"] / n, sums["noise_mse"] / n, sums["x0_mse"] / n
        print(f"Ep [{epoch:03d}/{epochs:03d}] | LR: {lr:.2e} | Noise_MSE: {tr_nmse:.4f} | x0_MSE: {tr_x0:.4f}")

        # 每 5 轮在验证集上执行真实 DDIM 逆向去噪采样，以真实生成表现选优保存检查点
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            reals, synths = [], []
            with torch.no_grad():
                for b_i, vb in enumerate(val_loader):
                    ve = vb["eeg_hrf" if use_hrf else "eeg"].to(device)
                    vf = vb["fnirs"].numpy()
                    sf = model.sample(ve, topo_ef, topo_fe, steps=25, corr_ef=corr_ef, spatial_affinity=spatial_affinity, seed=3000 + b_i).cpu().numpy()
                    reals.append(vf); synths.append(sf)
            reals = np.concatenate(reals, axis=0)
            synths = np.concatenate(synths, axis=0)
            val_mse = float(np.mean((reals - synths) ** 2))
            corrs = [
                float(((r - r.mean()) * (s - s.mean())).sum() /
                      (np.sqrt(((r - r.mean())**2).sum() * ((s - s.mean())**2).sum()) + 1e-12))
                for r, s in zip(reals, synths)
            ]
            val_spcc = float(np.mean(corrs))
            val_cpcc, _ = pearsonr(reals.mean(axis=(0, 1)), synths.mean(axis=(0, 1)))
            ddim_score = val_cpcc + 2.0 * val_spcc - 0.5 * val_mse

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([exp_id, epoch, f"{lr:.6e}", f"{tr_tot:.6f}", f"{tr_nmse:.6f}", f"{tr_x0:.6f}", f"{val_mse:.6f}", f"{val_spcc:.6f}", f"{val_cpcc:.6f}"])

            print(f"  --> [验证集 DDIM 采样 N={len(reals)}] MSE: {val_mse:.4f} | 单试次 PCC: {val_spcc:+.4f} | 曲线 PCC: {val_cpcc:+.4f}")

            if ddim_score > best_ddim_score:
                best_ddim_score = ddim_score
                torch.save({
                    "model": model.state_dict(),
                    "exp_id": exp_id,
                    "epoch": epoch,
                    "val_mse": val_mse,
                    "val_cpcc": val_cpcc
                }, os.path.join(cfg["train"]["checkpoint_dir"], f"scdm_{modality.lower()}_best.pt"))
                print(f"  ★ 已更新最佳 DDIM 生成模型 (Epoch {epoch} | Val_MSE: {val_mse:.4f}, Curve_PCC: {val_cpcc:+.4f})")

        if epoch == epochs:
            torch.save(model.state_dict(), os.path.join(cfg["train"]["checkpoint_dir"], f"scdm_{modality.lower()}_ep{epoch}.pt"))

if __name__ == "__main__":
    main()