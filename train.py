import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from data.shin_dataset import get_dataloaders
from models.scdm_unet import SCDMUnet
from models.diffusion import SCDMDiffusion
from utils.metrics import compute_generation_metrics

def main():
    with open("configs/shin_baseline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    modality = cfg["data"]["target_modality"]
    processed_dir = cfg["data"]["processed_dir"]

    topo_path = os.path.join(processed_dir, f"topo_{modality.lower()}.pt")
    topo_dict = torch.load(topo_path, map_location="cpu")
    topo_ef = topo_dict["topo_ef"].to(device)
    topo_fe = topo_dict["topo_fe"].to(device)

    train_loader, val_loader = get_dataloaders(
        processed_dir=processed_dir,
        modality=modality,
        batch_size=cfg["data"]["batch_size"],
        train_ratio=cfg["data"]["train_ratio"]
    )

    unet = SCDMUnet(time_dim=cfg["model"]["time_dim"]).to(device)
    model = SCDMDiffusion(
        unet,
        timesteps=cfg["model"]["timesteps"],
        beta_start=cfg["model"]["beta_start"],
        beta_end=cfg["model"]["beta_end"]
    ).to(device)

    # 降低初始学习率并加强权重衰减，抑制 10 轮后的平台震荡与过拟合
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=5e-6)

    best_score = -999.0
    print(f">>> 开始训练 SCDM ({modality}) | 训练集: {len(train_loader.dataset)} | 独立验证集: {len(val_loader.dataset)}")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        sums = {"total": 0.0, "mse": 0.0, "amp": 0.0, "pcc": 0.0, "smooth": 0.0}
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{cfg['train']['epochs']}]", leave=False)

        for batch in pbar:
            e_0 = batch["eeg"].to(device)
            f_0 = batch["fnirs"].to(device)

            optimizer.zero_grad()
            loss, l_dict = model.compute_loss(e_0, f_0, topo_ef, topo_fe)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in sums:
                sums[k] += l_dict[k]
            pbar.set_postfix({"mse": f"{l_dict['mse']:.3f}", "pcc_l": f"{l_dict['pcc']:.3f}"})

        scheduler.step()
        n = len(train_loader)
        print(
            f"Ep [{epoch:03d}] | Total: {sums['total']/n:.4f} "
            f"(MSE: {sums['mse']/n:.4f} | Amp: {sums['amp']/n:.4f} | "
            f"PCC_L: {sums['pcc']/n:.4f} | Smooth: {sums['smooth']/n:.4f})"
        )

        # 每 5 轮在独立验证集上评估一次，精准捕捉最佳泛化点
        if epoch % 5 == 0 or epoch == cfg["train"]["epochs"]:
            model.eval()
            all_real, all_syn = [], []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_e = val_batch["eeg"].to(device)
                    val_f = val_batch["fnirs"].numpy()
                    syn_f = model.sample(val_e, topo_ef, topo_fe, ddim_steps=20).cpu().numpy()
                    all_real.append(val_f)
                    all_syn.append(syn_f)

            all_real = np.concatenate(all_real, axis=0)
            all_syn = np.concatenate(all_syn, axis=0)
            metrics = compute_generation_metrics(all_real, all_syn)
            rmse, pcc = metrics["RMSE"], metrics["Curve_PCC"]
            # 综合评价指标：兼顾高 PCC 与低 RMSE
            score = pcc - 0.5 * rmse
            print(f"  --> [独立验证集 N={len(all_real)}] RMSE: {rmse:.4f} | 均值曲线 PCC: {pcc:.4f}")

            if score > best_score:
                best_score = score
                best_path = os.path.join(cfg["train"]["checkpoint_dir"], f"scdm_{modality.lower()}_best.pt")
                torch.save(model.state_dict(), best_path)
                print(f"  ★ 刷新最优权重 (RMSE: {rmse:.4f}, PCC: {pcc:.4f}) -> {best_path}")

            if epoch % cfg["train"]["save_interval"] == 0:
                ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], f"scdm_{modality.lower()}_ep{epoch}.pt")
                torch.save(model.state_dict(), ckpt_path)

if __name__ == "__main__":
    main()