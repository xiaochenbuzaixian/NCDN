import os
import yaml
import torch
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

    # 1. 加载空间拓扑图
    topo_path = os.path.join(processed_dir, f"topo_{modality.lower()}.pt")
    topo_dict = torch.load(topo_path, map_location=device)
    topo_ef = topo_dict["topo_ef"].to(device)
    topo_fe = topo_dict["topo_fe"].to(device)

    # 2. 加载数据集
    train_loader, val_loader = get_dataloaders(
        processed_dir=processed_dir,
        modality=modality,
        batch_size=cfg["data"]["batch_size"],
        train_ratio=cfg["data"]["train_ratio"]
    )

    # 3. 初始化模型与优化器
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

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch:03d}/{cfg['train']['epochs']:03d}] | Train MSE Loss: {avg_loss:.4f}")

        # 定期保存权重并抽样验证
        if epoch % cfg["train"]["save_interval"] == 0:
            ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], f"scdm_{modality.lower()}_ep{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)

            model.eval()
            val_batch = next(iter(val_loader))
            val_e = val_batch["eeg"][:4].to(device)
            val_f = val_batch["fnirs"][:4].to(device)
            syn_f = model.sample(val_e, topo_ef, topo_fe)
            metrics = compute_generation_metrics(val_f, syn_f)
            print(f"  --> [验证评估] RMSE: {metrics['RMSE']:.4f} | 均值曲线 PCC: {metrics['Curve_PCC']:.4f}")

if __name__ == "__main__":
    main()