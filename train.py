import os
import math
import yaml
import torch
import numpy as np
from tqdm import tqdm
from data.shin_dataset import get_dataloaders
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion
from scipy.stats import pearsonr

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

def cosine_lr(epoch, warmup, total, peak_lr, min_lr=1e-5):
    if epoch < warmup:
        return min_lr + (peak_lr - min_lr) * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def compute_eval_metrics(real, synth):
    mse = float(np.mean((real - synth) ** 2))
    rmse = float(np.sqrt(mse))
    sample_corrs = []
    for r, s in zip(real, synth):
        rf, sf = r.ravel() - r.mean(), s.ravel() - s.mean()
        denom = np.sqrt((rf ** 2).sum() * (sf ** 2).sum()) + 1e-12
        sample_corrs.append(float((rf * sf).sum() / denom))
    curve_pcc, _ = pearsonr(real.mean(axis=(0, 1)), synth.mean(axis=(0, 1)))
    return {"MSE": mse, "RMSE": rmse, "Sample_PCC": float(np.mean(sample_corrs)), "Curve_PCC": float(curve_pcc)}

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
    topo_ef = topo_dict["topo_ef"].to(device)
    topo_fe = topo_dict["topo_fe"].to(device)
    topo_ee = topo_dict.get("topo_ee", topo_ef).to(device)
    topo_ff = topo_dict.get("topo_ff", topo_fe).to(device)

    train_loader, val_loader = get_dataloaders(
        processed_dir=processed_dir,
        modality=modality,
        batch_size=cfg["data"]["batch_size"],
        train_ratio=cfg["data"]["train_ratio"]
    )

    unet = SCDMUnet(
        time_dim=cfg["model"]["time_dim"],
        base_channels=cfg["model"].get("base_channels", 128)
    ).to(device)
    model = SCDMDiffusion(
        unet,
        timesteps=cfg["model"]["timesteps"],
        beta_start=cfg["model"]["beta_start"],
        beta_end=cfg["model"]["beta_end"],
        schedule=cfg["model"].get("schedule", "cosine")
    ).to(device)

    ema = EMA(model, decay=0.995)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=1e-4
    )

    epochs = cfg["train"]["epochs"]
    warmup = cfg["train"].get("warmup_epochs", 15)
    peak_lr = cfg["train"]["lr"]
    best_score = -999.0

    print(f">>> 启动 Prior-Guided SCDM 训练 | 设备: {device} | 训练集: {len(train_loader.dataset)} | 验证集: {len(val_loader.dataset)}")

    for epoch in range(1, epochs + 1):
        lr = cosine_lr(epoch - 1, warmup, epochs, peak_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        sums = {"total": 0.0, "noise_mse": 0.0, "x0_mse": 0.0, "pcc_loss": 0.0}
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:03d}/{epochs:03d}]", leave=False)

        for batch in pbar:
            e_in = batch["eeg_hrf" if use_hrf else "eeg"].to(device)
            f_0 = batch["fnirs"].to(device)

            optimizer.zero_grad()
            loss, l_dict = model.compute_loss(e_in, f_0, topo_ef, topo_fe, topo_ee, topo_ff)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            for k in sums:
                sums[k] += l_dict[k]
            pbar.set_postfix({"x0_mse": f"{l_dict['x0_mse']:.4f}", "pcc_l": f"{l_dict['pcc_loss']:.4f}"})

        n = len(train_loader)
        print(
            f"Ep [{epoch:03d}/{epochs:03d}] | LR: {lr:.2e} | Noise_MSE: {sums['noise_mse']/n:.4f} "
            f"| Prior_MSE: {sums['x0_mse']/n:.4f} | Train_PCC: {1.0 - sums['pcc_loss']/n:.4f}"
        )

        # 每 10 轮在独立验证集上评估并基于综合生成质量保存 best.pt
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            all_real, all_syn = [], []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_e = val_batch["eeg_hrf" if use_hrf else "eeg"].to(device)
                    val_f = val_batch["fnirs"].numpy()
                    syn_f = model.sample(val_e, topo_ef, topo_fe, topo_ee, topo_ff, steps=25).cpu().numpy()
                    all_real.append(val_f)
                    all_syn.append(syn_f)

            all_real = np.concatenate(all_real, axis=0)
            all_syn = np.concatenate(all_syn, axis=0)
            m = compute_eval_metrics(all_real, all_syn)
            score = m["Sample_PCC"] + 0.5 * m["Curve_PCC"] - 0.5 * m["MSE"]
            print(
                f"  --> [验证集 N={len(all_real)}] MSE: {m['MSE']:.4f} (RMSE: {m['RMSE']:.4f}) "
                f"| 单试次 PCC: {m['Sample_PCC']:.4f} | 均值曲线 PCC: {m['Curve_PCC']:.4f}"
            )

            if score > best_score:
                best_score = score
                best_path = os.path.join(cfg["train"]["checkpoint_dir"], f"scdm_{modality.lower()}_best.pt")
                torch.save({"model": model.state_dict(), "ema": ema.shadow, "epoch": epoch, "loss": m["MSE"]}, best_path)
                print(f"  ★ 刷新最佳模型 (Epoch {epoch} | MSE: {m['MSE']:.4f} | 曲线 PCC: {m['Curve_PCC']:.4f})")

            if epoch % cfg["train"]["save_interval"] == 0:
                ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], f"scdm_{modality.lower()}_ep{epoch}.pt")
                torch.save({"model": model.state_dict(), "ema": ema.shadow, "epoch": epoch, "loss": m["MSE"]}, ckpt_path)

if __name__ == "__main__":
    main()