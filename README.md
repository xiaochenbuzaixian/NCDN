# NCDN: Neurovascular-guided Conditional Diffusion Network for Cross-Modal EEG-to-fNIRS Generation

<p align="center">
  <b><a href="#中文版本">简体中文</a> | <a href="#english-version">English</a></b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Dataset-Shin2017%20EEG--fNIRS-success.svg" alt="Dataset">
  <img src="https://img.shields.io/badge/Task-Motor%20Imagery%20%26%20Cross--Modal%20Synthesis-orange.svg" alt="Task">
</p>

---

<a name="中文版本"></a>
## 🇨🇳 简体中文版本

### 1. 项目简介
**NCDN（Neurovascular-guided Conditional Diffusion Network）** 是一个用于跨模态脑电-近红外（EEG-to-fNIRS）神经血管信号合成与多模态脑机接口（BCI）解码的深度学习框架。

本项目基于条件扩散生成模型（**SCDM**），仅利用高时间分辨率的头皮 EEG 信号，即可高保真地跨模态合成双血氧 fNIRS（HbO 与 HbR）波形，克服了传统 fNIRS 硬件佩戴繁琐的缺陷。同时，利用特征引导注意力融合网络（**FGANet**），在零数据泄漏的严格被试内（Subject-Dependent）评估协议下，显著提升了运动想象（Motor Imagery, MI）的解码性能。

### 2. 硬件要求与运行环境
本项目基于 PyTorch 开发，支持 GPU 加速训练与推理。
* **操作系统**: Linux (推荐 Ubuntu 20.04/22.04) / Windows 11
* **GPU**: 推荐单张 NVIDIA 显卡 (显存 $\ge$ 12GB，如 RTX 3090 / 4090 / A100 / RTX 4070Ti)
* **CUDA**: 11.7 及以上版本
* **核心依赖**:
  ```text
  python >= 3.9
  torch >= 2.0.0
  numpy, scipy, scikit-learn, matplotlib, pyyaml, tqdm, dcor
  ```

### 3. 项目文件结构
```text
NCDN/
├── checkpoints/                  # 训练生成的模型权重目录 (.pt)
├── configs/
│   └── shin_baseline.yaml        # 数据集、扩散模型与训练超参数配置文件
├── data/
│   ├── preprocess_shin.py        # Shin2017 数据集预处理 (MBLL解算、带通滤波、HRF提取)
│   ├── shin_dataset.py           # PyTorch Dataset 与 DataLoader 封装
│   └── topology_builder.py       # 16x16 头皮空间网格映射与跨模态相关性平面构建
├── datasets/
│   ├── raw/                      # 原始 Shin2017 数据集存放位置 (29名被试, EEG & fNIRS)
│   └── processed/                # 预处理生成的零泄漏张量与划分索引缓存 (.pt / .json)
├── models/
│   ├── modules/                  # 扩散网络底层组件 (SCG空间模块, MTR时序模块等)
│   ├── scdm_unet.py              # 6-Block SCDM U-Net 扩散去噪主干网络
│   └── diffusion.py              # v-prediction 条件扩散调度与 DDIM 逆向采样
├── outputs/                      # 评估指标 CSV 表格与论文级可视化图表输出目录
├── train.py                      # 一键训练完整 SCDM 主模型与 4 组消融变体
└── evaluate.py                   # 一键执行 DDIM 采样、绘制波形/拓扑图、运行分类评估
```

### 4. 快速开始 (Quick Start)

#### 第一步：克隆代码与配置环境
```bash
git clone https://github.com/xiaochenbuzaixian/NCDN.git
cd NCDN
conda create -n ncdn python=3.9 -y
conda activate ncdn
pip install -r requirements.txt
```

#### 第二步：数据集准备
1. 下载 Shin et al. (2017) 运动想象数据集；
2. 将原始文件解压并放置于 `datasets/raw/` 目录下，确保结构为 `datasets/raw/EEG/subject 01/...` 与 `datasets/raw/fNIRS/subject 01/...`。

#### 第三步：数据预处理与特征提取
本步骤将提取双伽马神经血管包络，并基于固定的 `70% 训练集` 构建跨模态拓扑矩阵，确保 **100% 零测试集泄漏**。
```bash
python data/preprocess_shin.py
python data/topology_builder.py
```

#### 第四步：模型训练
运行以下命令，系统将自动从头训练 72 通道双血氧联合扩散主模型（SCDM）以及完整的 4 组消融实验变体：
```bash
# 默认使用 0 号 GPU，可根据实际情况修改 CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0 python train.py --ablation
```

#### 第五步：模型评估与可视化
运行评估脚本，自动执行 DDIM 采样、特征提取、被试内（Subject-Dependent）分类以及图表绘制：
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py
```
运行完成后，请检查 `outputs/` 目录获取完整的 CSV 结果表与论文级高清配图（双波峰血流动力学响应图、空间映射图等）。

### 5. 核心实验结果

#### 5.1 多模态运动想象分类实验 (Subject-Dependent 评估)
在 Shin2017 数据集（29 名被试，1740 个试次）上，5 组核心实验严格呈现单调递进规律。采用 FGANet 结合合成 fNIRS，准确率较单模态 EEG 提升显著：
* **EEG-only**: 71.65%
* **EEG + Synth HbO (SCDM)**: 73.95% (较 EEG 提升 **+2.30%**)
* **EEG + Real HbO**: 74.33% (较 EEG 提升 **+2.68%**)

#### 5.2 核心模块消融实验
证明了跨模态空间模块（**SCG**）与多尺度时序模块（**MTR**）的有效性，完整 SCDM 逼近单模态生成上限：
* **ATTN + COV (Baseline)**: ~70.88%
* **SCG(EEG) + MTR (Ours)**: ~73.95%
* **SCG(fNIRS) + MTR (Upper bound)**: ~74.71%

---

<a name="english-version"></a>
## 🌐 English Version

### 1. Overview
**NCDN (Neurovascular-guided Conditional Diffusion Network)** is a deep learning framework designed for cross-modal EEG-to-fNIRS neurovascular signal synthesis and multimodal Brain-Computer Interface (BCI) decoding. 

By leveraging the **Spatio-Temporal Cross-Diffusion Model (SCDM)**, this project synthesizes high-fidelity, dual-hemoglobin fNIRS waveforms (HbO and HbR) directly from high-temporal-resolution scalp EEG signals. Combined with a Feature-Guided Attention Network (**FGANet**), it significantly enhances Motor Imagery (MI) decoding performance under a strict, zero-leakage, subject-dependent evaluation protocol.

### 2. Hardware & Environment Requirements
* **OS**: Linux (Ubuntu 20.04/22.04 recommended) / Windows 11
* **GPU**: A single NVIDIA GPU with $\ge$ 12GB VRAM (e.g., RTX 3090 / 4090 / A100)
* **CUDA**: Version 11.7 or higher
* **Dependencies**: Python $\ge$ 3.9, PyTorch $\ge$ 2.0.0

### 3. Quick Start

**Step 1: Setup Environment**
```bash
git clone https://github.com/xiaochenbuzaixian/NCDN.git
cd NCDN
conda create -n ncdn python=3.9 -y
conda activate ncdn
pip install -r requirements.txt
```

**Step 2: Prepare Dataset**
Place the raw Shin et al. (2017) dataset into the `datasets/raw/` directory.

**Step 3: Preprocessing & Topology Building**
```bash
python data/preprocess_shin.py
python data/topology_builder.py
```

**Step 4: Train the Models**
Train the main 72-channel dual-hemoglobin SCDM and all ablation variants:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --ablation
```

**Step 5: Evaluate & Visualize**
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py
```
Check the `outputs/` folder for comprehensive CSV result tables and publication-ready figures (e.g., bimodal hemodynamic response curves, spatial mapping topologies).