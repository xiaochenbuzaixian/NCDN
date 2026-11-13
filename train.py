import os
import json
import math
import copy
import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion

def cosine_lr(epoch, warmup, total, peak_lr, min_lr=1e-5):
    if epoch < warmup:
        return min_lr + (peak_lr - min_lr) * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def build_dual_dataloaders(processed_dir, batch_size=16, use_hrf=True):
    eeg_file = "eeg_hrf_epochs.pt" if use_hrf else "eeg_epochs.pt"
    eeg = torch.load(os.path.join(processed_dir, eeg_file)).float()
    hbr = torch.load(os.path.join(processed_dir, "fnirs_hbr_epochs.pt")).float()
    hbo = torch.load(os.path.join(processed_dir, "fnirs_hbo_epochs.pt")).float()
    fnirs_dual = torch.cat([hbr, hbo], dim=1)  # (1740, 72, 256)

    with open(os.path.join(processed_dir, "split_indices.json"), "r") as f:
        sp = json.load(f)
    tr_idx = torch.tensor(sp["train_indices"], dtype=torch.long)
    val_idx = torch.tensor(sp["val_indices"], dtype=torch.long)

    train_ds = TensorDataset(eeg[tr_idx], fnirs_dual[tr_idx])
    val_ds = TensorDataset(eeg[val_idx], fnirs_dual[val_idx])
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False))

def train_variant(variant_code, save_filename, epochs, val_interval, train_loader, val_loader,
                  topo_ef, topo_fe, topo_ff, corr_ef, spatial_affinity, cfg, device):
    unet = SCDMUnet(
        time_dim=cfg["model"].get("time_dim", 64),
        base_channels=cfg["model"].get("base_channels", 32),
        n_blocks=6,
        out_channels=72,
        variant=variant_code
    ).to(device)
    model = SCDMDiffusion(unet, timesteps=cfg["model"]["timesteps"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)

    warmup = min(12, epochs // 6)
    peak_lr = cfg["train"]["lr"]
    best_score = -999.0

    print("\n" + "=" * 88)
    print(f">>> 启动训练 [{variant_code}] (同相双峰 2×SCG + 2×MTR 交替架构) -> {save_filename} | 轮数: {epochs}")
    print("=" * 88)

    for epoch in range(1, epochs + 1):
        lr = cosine_lr(epoch - 1, warmup, epochs, peak_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        pbar = tqdm(train_loader, desc=f"[{variant_code}] Ep {epoch:03d}/{epochs:03d}", leave=False)
        for e_in, f_dual in pbar:
            e_in, f_dual = e_in.to(device), f_dual.to(device)
            optimizer.zero_grad()
            loss, l_dict = model.compute_loss(
                e_in, f_dual, topo_ef, topo_fe, corr_ef=corr_ef, spatial_affinity=spatial_affinity, topo_ff=topo_ff
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pbar.set_postfix({"n_mse": f"{l_dict['noise_mse']:.4f}", "x0_mse": f"{l_dict['x0_mse']:.4f}"})

        if epoch % val_interval == 0 or epoch == epochs:
            model.eval()
            reals, synths = [], []
            with torch.no_grad():
                for b_i, (ve, vf) in enumerate(val_loader):
                    ve = ve.to(device)
                    vf_dev = vf.to(device) if variant_code == "SCG_fNIRS_MTR" else None
                    sf = model.sample(
                        ve, topo_ef, topo_fe, steps=20, corr_ef=corr_ef,
                        spatial_affinity=spatial_affinity, topo_ff=topo_ff, f_self_cond=vf_dev, seed=3000 + b_i
                    ).cpu().numpy()
                    reals.append(vf.numpy()); synths.append(sf)
            reals = np.concatenate(reals, axis=0)
            synths = np.concatenate(synths, axis=0)

            r_hbr, s_hbr = reals[:, :36, :], synths[:, :36, :]
            r_hbo, s_hbo = reals[:, 36:, :], synths[:, 36:, :]

            mse_hbr = float(np.mean((r_hbr - s_hbr) ** 2))
            mse_hbo = float(np.mean((r_hbo - s_hbo) ** 2))
            cpcc_hbr, _ = pearsonr(r_hbr.mean(axis=(0, 1)), s_hbr.mean(axis=(0, 1)))
            cpcc_hbo, _ = pearsonr(r_hbo.mean(axis=(0, 1)), s_hbo.mean(axis=(0, 1)))

            score = cpcc_hbr + cpcc_hbo - 0.3 * (mse_hbr + mse_hbo)
            print(f"Ep [{epoch:03d}/{epochs:03d}] | Val HbR (MSE:{mse_hbr:.4f}, PCC:{cpcc_hbr:+.4f}) | Val HbO (MSE:{mse_hbo:.4f}, PCC:{cpcc_hbo:+.4f})")

            if score > best_score:
                best_score = score
                save_path = os.path.join(cfg["train"]["checkpoint_dir"], save_filename)
                torch.save({
                    "model": copy.deepcopy(model.state_dict()),
                    "variant": variant_code,
                    "epoch": epoch,
                    "cpcc_hbr": cpcc_hbr,
                    "cpcc_hbo": cpcc_hbo
                }, save_path)
                print(f"  ★ 已更新最佳模型 -> {save_filename} (HbR_PCC: {cpcc_hbr:+.4f}, HbO_PCC: {cpcc_hbo:+.4f})")

def main():
    with open("configs/shin_baseline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    ckpt_dir = cfg["train"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    processed_dir = cfg["data"]["processed_dir"]
    use_hrf = cfg["data"].get("eeg_hrf", True)

    topo_hbr = torch.load(os.path.join(processed_dir, "topo_hbr.pt"), map_location="cpu")
    topo_hbo = torch.load(os.path.join(processed_dir, "topo_hbo.pt"), map_location="cpu")
    topo_ef = (0.5 * (topo_hbr["topo_ef"] + topo_hbo["topo_ef"])).to(device)
    topo_fe = (0.5 * (topo_hbr["topo_fe"] + topo_hbo["topo_fe"])).to(device)
    topo_ff = (0.5 * (topo_hbr["topo_ff"] + topo_hbo["topo_ff"])).to(device)
    corr_ef = (0.5 * (topo_hbr["corr_ef"] + topo_hbo["corr_ef"])).to(device)
    spatial_affinity = topo_hbr["spatial_affinity"].to(device)

    train_loader, val_loader = build_dual_dataloaders(
        processed_dir, batch_size=cfg["data"]["batch_size"], use_hrf=use_hrf
    )

    # 1. 训练完整主模型: 5. SCG(EEG) + MTR (Full SCDM, 120 Epochs)
    train_variant(
        "SCG_EEG_MTR", "scdm_dual_best.pt", epochs=120, val_interval=20,
        train_loader=train_loader, val_loader=val_loader,
        topo_ef=topo_ef, topo_fe=topo_fe, topo_ff=topo_ff, corr_ef=corr_ef, spatial_affinity=spatial_affinity,
        cfg=cfg, device=device
    )
    dual_p = os.path.join(ckpt_dir, "scdm_dual_best.pt")
    torch.save(torch.load(dual_p), os.path.join(ckpt_dir, "scdm_hbr_best.pt"))

    # 2. 依次训练其余 4 组原论文消融实验变体 (各 60 Epochs)
    ablation_plan = [
        ("ATTN_COV",      "scdm_abl_1_attn_cov.pt",      60),
        ("ATTN_MTR",      "scdm_abl_2_attn_mtr.pt",      60),
        ("SCG_fNIRS_MTR", "scdm_abl_3_scg_fnirs_mtr.pt", 60),
        ("SCG_EEG_COV",   "scdm_abl_4_scg_eeg_cov.pt",   60),
    ]
    for v_code, v_file, ep_num in ablation_plan:
        train_variant(
            v_code, v_file, epochs=ep_num, val_interval=20,
            train_loader=train_loader, val_loader=val_loader,
            topo_ef=topo_ef, topo_fe=topo_fe, topo_ff=topo_ff, corr_ef=corr_ef, spatial_affinity=spatial_affinity,
            cfg=cfg, device=device
        )

    print("\n✅ 全部 5 套模型权重重新训练完成！现在请运行 `CUDA_VISIBLE_DEVICES=1 python evaluate.py` 查看结果。")

if __name__ == "__main__":
    main()