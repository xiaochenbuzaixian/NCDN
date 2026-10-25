def main():
    raw_dir = "datasets/raw"
    save_dir = "datasets/processed"
    os.makedirs(save_dir, exist_ok=True)

    all_eeg, all_hbo, all_hbr, all_labels, all_subs = [], [], [], [], []

    print(">>> 开始预处理 Shin2017 数据集 (含任务前 [-5s, 0s] 基线归零校正)...")
    for sub in tqdm(range(1, 30)):
        e, hbo, hbr, l, s = process_single_subject(sub, raw_dir)
        all_eeg.append(e)
        all_hbo.append(hbo)
        all_hbr.append(hbr)
        all_labels.append(l)
        all_subs.append(s)

    all_eeg = torch.tensor(np.concatenate(all_eeg, axis=0))
    all_hbo = torch.tensor(np.concatenate(all_hbo, axis=0))
    all_hbr = torch.tensor(np.concatenate(all_hbr, axis=0))
    all_labels = torch.tensor(np.concatenate(all_labels, axis=0))
    all_subs = torch.tensor(np.concatenate(all_subs, axis=0))

    # 1. EEG: 去基线后按通道全局标准化
    all_eeg = all_eeg - all_eeg[:, :, :800].mean(dim=-1, keepdim=True)
    eeg_std = all_eeg.std(dim=(0, -1), keepdim=True).clamp(min=1e-4)
    all_eeg = torch.nan_to_num(all_eeg / eeg_std, 0.0)

    # 2. fNIRS (HbO / HbR): 严格使用任务前 [-5s, 0s] (前50个点) 进行基线归零校正！
    # 彻底解决整段减均值导致的基线上下漂移问题
    def baseline_correct_and_norm(fnirs_tensor):
        base = fnirs_tensor[:, :, :50].mean(dim=-1, keepdim=True)
        corrected = fnirs_tensor - base
        # 使用全局通道标准差缩放，保留高激活通道与静息通道的生理幅值比
        global_std = corrected.std(dim=(0, -1), keepdim=True).clamp(min=1e-4)
        normed = corrected / global_std
        return torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)

    all_hbo = baseline_correct_and_norm(all_hbo)
    all_hbr = baseline_correct_and_norm(all_hbr)

    print(f"\n>>> 处理完成！汇总形状与基线检查:")
    print(f"  EEG: {all_eeg.shape} | 均值: {all_eeg.mean():.4f}, 标准差: {all_eeg.std():.4f}")
    print(f"  HbO: {all_hbo.shape} | [-5s,0s]基线均值: {all_hbo[:, :, :50].mean():.6f}, 全局标准差: {all_hbo.std():.4f}")
    print(f"  HbR: {all_hbr.shape} | [-5s,0s]基线均值: {all_hbr[:, :, :50].mean():.6f}, 全局标准差: {all_hbr.std():.4f}")

    torch.save(all_eeg, os.path.join(save_dir, "eeg_epochs.pt"))
    torch.save(all_hbo, os.path.join(save_dir, "fnirs_hbo_epochs.pt"))
    torch.save(all_hbr, os.path.join(save_dir, "fnirs_hbr_epochs.pt"))
    torch.save(all_labels, os.path.join(save_dir, "labels.pt"))
    torch.save(all_subs, os.path.join(save_dir, "subject_ids.pt"))
    print(f">>> 归零校准后的张量已保存至 {save_dir}/")