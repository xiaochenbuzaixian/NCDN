import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 模拟你的平滑基线数据 (将时间轴从 -5~20 平移至 0~25)
# ==========================================
t = np.linspace(0, 25, 500)

# 按照你复现的曲线走势构建平滑基线 (放缩到原论文的量级)
# HBR LMI
real_hbr_lmi = np.sin(t / 3 - 1) * 3000 + np.sin(t / 1.5) * 1500 + 2000
synth_hbr_lmi = np.sin(t / 3 - 1.2) * 4500 + np.sin(t / 1.5) * 1000 + 3000

# HBR RMI
real_hbr_rmi = -np.cos(t / 3.5) * 7000 + 1000
synth_hbr_rmi = -np.cos(t / 3.5 - 0.2) * 6500 + 500

# HBO LMI
real_hbo_lmi = -np.sin(t / 2.5) * 2000 + t * 100
synth_hbo_lmi = -np.sin(t / 2.5) * 1500 - 1000

# HBO RMI
real_hbo_rmi = np.sin(t / 3 - 0.5) * 2000 + 1000
synth_hbo_rmi = np.sin(t / 3 - 0.7) * 2500 + 500

# ==========================================
# 2. 注入高频“毛刺” (模拟真实/合成的未平滑时序噪声)
# ==========================================
np.random.seed(42)  # 保证每次生成的毛刺形状一致
# 根据不同信号的幅值设定噪声强度
noise_hbr = 800
noise_hbo = 300

# 将平滑曲线加上高斯噪声
real_hbr_lmi += np.random.normal(0, noise_hbr, len(t))
synth_hbr_lmi += np.random.normal(0, noise_hbr, len(t))

real_hbr_rmi += np.random.normal(0, noise_hbr, len(t))
synth_hbr_rmi += np.random.normal(0, noise_hbr, len(t))

real_hbo_lmi += np.random.normal(0, noise_hbo, len(t))
synth_hbo_lmi += np.random.normal(0, noise_hbo, len(t))

real_hbo_rmi += np.random.normal(0, noise_hbo, len(t))
synth_hbo_rmi += np.random.normal(0, noise_hbo, len(t))

# ==========================================
# 3. 开始按照原论文风格绘图
# ==========================================
# 设置全局字体样式 (类似原论文的衬线字体风格)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, axs = plt.subplots(2, 2, figsize=(10, 9))
plt.subplots_adjust(wspace=0.25, hspace=0.3)

# 颜色定义 (严格对齐原论文: 蓝+橙)
color_real = '#1f77b4'  # 真实信号蓝色
color_synth = '#ff7f0e'  # 合成信号橙色 (如果你坚持要红色实线，可改为 'red')

# ----------------- HBR LMI (左上) -----------------
axs[0, 0].plot(t, real_hbr_lmi, color=color_real, linewidth=2, label='real')
axs[0, 0].plot(t, synth_hbr_lmi, color=color_synth, linewidth=2, label='synthetic')  # 变实线
axs[0, 0].set_title('HBR LMI', fontweight='bold', pad=15)
axs[0, 0].set_ylabel('$\mu$M', fontsize=12, fontweight='bold')
axs[0, 0].set_ylim(-6000, 12000)
axs[0, 0].set_yticks([-5000, 0, 5000, 10000])

# ----------------- HBR RMI (右上) -----------------
axs[0, 1].plot(t, real_hbr_rmi, color=color_real, linewidth=2, label='real')
axs[0, 1].plot(t, synth_hbr_rmi, color=color_synth, linewidth=2, label='synthetic')
axs[0, 1].set_title('HBR RMI', fontweight='bold', pad=15)
axs[0, 1].set_ylim(-12000, 10000)
axs[0, 1].set_yticks([-10000, -5000, 0, 5000, 10000])

# ----------------- HBO LMI (左下) -----------------
axs[1, 0].plot(t, real_hbo_lmi, color=color_real, linewidth=2, label='real')
axs[1, 0].plot(t, synth_hbo_lmi, color=color_synth, linewidth=2, label='synthetic')
axs[1, 0].set_title('HBO LMI', fontweight='bold', pad=15)
axs[1, 0].set_ylabel('$\mu$M', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('Time (s)', fontsize=12)
axs[1, 0].set_ylim(-3500, 4500)
axs[1, 0].set_yticks([-2000, 0, 2000, 4000])

# ----------------- HBO RMI (右下) -----------------
axs[1, 1].plot(t, real_hbo_rmi, color=color_real, linewidth=2, label='real')
axs[1, 1].plot(t, synth_hbo_rmi, color=color_synth, linewidth=2, label='synthetic')
axs[1, 1].set_title('HBO RMI', fontweight='bold', pad=15)
axs[1, 1].set_xlabel('Time (s)', fontsize=12)
axs[1, 1].set_ylim(-2500, 5000)
axs[1, 1].set_yticks([-2000, 0, 2000, 4000])

# ==========================================
# 4. 全局样式统一处理 (去除多余边框、增加基准线)
# ==========================================
for ax in axs.flat:
    # X 轴统一设置
    ax.set_xlim(0, 25.5)
    ax.set_xticks([0, 10, 20])
    ax.tick_params(axis='both', labelsize=11)

    # 隐藏上方和右侧的边框线 (Spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 加粗左侧和下侧边框
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # 添加 y=0 的黑色基准实线
    ax.axhline(0, color='black', linewidth=1)

    # 图例样式匹配原论文 (放在左上角，保留外框)
    ax.legend(loc='upper left', fontsize=11, frameon=True, edgecolor='lightgray')

plt.show()