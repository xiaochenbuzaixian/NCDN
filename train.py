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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"], eta_min=1e-5)

    best_pcc = -1.0
    print(f">>> 开始训练 SCDM ({modality}) | 设备: {device} | 总轮数: {cfg['train']['epochs']}")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{cfg['train']['epochs']}]", leave=False)

        for batch in pbar:
            e_0 = batch["eeg"].to(device)
            f_0 = batch["fnirs"].to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(e_0, f_0, topo_ef, topo_fe)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch:03d}/{cfg['train']['epochs']:03d}] | Train Joint Loss: {avg_loss:.4f}")

        # 每 10 轮在完整验证集上进行一次快速评估并保存最优权重
        if epoch % 10 == 0 or epoch == cfg["train"]["epochs"]:
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
            print(f"  --> [全集验证] RMSE: {metrics['RMSE']:.4f} | 均值曲线 PCC: {metrics['Curve_PCC']:.4f}")

            if metrics["Curve_PCC"] > best_pcc:
                best_pcc = metrics["Curve_PCC"]
                best_path = os.path.join(cfg["train"]["checkpoint_dir"], f"scdm_{modality.lower()}_best.pt")
                torch.save(model.state_dict(), best_path)
                print(f"  ★ 已刷新最优模型 (Best PCC: {best_pcc:.4f}) -> {best_path}")

            if epoch % cfg["train"]["save_interval"] == 0:
                ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], f"scdm_{modality.lower()}_ep{epoch}.pt")
                torch.save(model.state_dict(), ckpt_path)

if __name__ == "__main__":
    main()