# ERP Benchmark 复现环境搭建指南

## Mac M2 本地环境（Apple Silicon）

### 1. 创建独立虚拟环境（强烈推荐）

```bash
# 用 conda 管理（推荐，对 M2 芯片支持更好）
conda create -n erp_bench python=3.10
conda activate erp_bench

# 或者用 venv
python3 -m venv erp_bench
source erp_bench/bin/activate
```

### 2. 安装核心依赖（M2 专用顺序）

```bash
# 先装 PyTorch（M2 原生 MPS 加速版）
pip install torch torchvision torchaudio

# 验证 MPS（M2 GPU）是否可用
python -c "import torch; print(torch.backends.mps.is_available())"
# 应输出 True

# EEG 信号处理核心库
pip install mne==1.6.1

# 数据科学基础
pip install numpy scipy scikit-learn pandas matplotlib seaborn

# 其他工具
pip install tqdm einops h5py
```

### 3. 拉取官方代码

```bash
git clone https://github.com/DL4mHealth/ERP-Benchmark.git
cd ERP-Benchmark
```

### 4. 验证安装（运行这个脚本）

```python
# verify_env.py
import torch
import mne
import numpy as np
import sklearn

print("=" * 40)
print("环境验证")
print("=" * 40)
print(f"PyTorch:    {torch.__version__}")
print(f"MNE:        {mne.__version__}")
print(f"NumPy:      {np.__version__}")
print(f"Scikit:     {sklearn.__version__}")
print(f"MPS 可用:   {torch.backends.mps.is_available()}")  # M2 GPU
print(f"CUDA 可用:  {torch.cuda.is_available()}")

# 模拟一个ERP数据张量
# shape = (trials, timepoints, channels)
fake_erp = torch.randn(32, 200, 64)  # 32个trial, 200个时间点, 64个通道
print(f"\n模拟ERP数据: {fake_erp.shape}")
print("  - 32   = batch中的trial数量")
print("  - 200  = 时间点 (200Hz × 1秒)")
print("  - 64   = EEG通道数")
print("\n✓ 环境验证通过！")
```

```bash
python verify_env.py
```

---

## 理解数据格式（最关键）

ERP数据本质上是一个3D张量：

```
数据形状: (N_trials, T_timepoints, C_channels)

以 CESCA-AODD 为例:
- N = 38,151 个trial（刺激事件）
- T = 200 个时间点（200Hz采样率 × 1秒）
- C = 26 个EEG通道

标签: (N_trials,) → 0=标准刺激, 1=目标刺激
```

---

## 数据获取（12个数据集来源）

| 数据集 | 来源 | 大小 | 优先级 |
|--------|------|------|--------|
| CESCA-AODD/VODD/FLANKER | OpenNeuro | 中等 | ⭐ 先下这个 |
| mTBI-ODD | OpenNeuro | 中等 | 二优先 |
| PD-SIM / PD-ODD | OpenNeuro | 较大 | 三优先 |

```bash
# 安装 OpenNeuro 下载工具
pip install openneuro-py

# 下载 CESCA 数据集（论文中最多subjects的数据集）
# 具体 accession number 需查看论文 GitHub 仓库的 README
```

---

## 推荐复现顺序

### Phase 1（本地 Mac，1-2天）
目标：跑通完整 pipeline

```
数据集: CESCA-AODD（最大、最标准）
模型:   EEGNet（最轻量，< 10K 参数）
验证:   能跑出 Accuracy/F1/AUROC 三个指标
```

### Phase 2（本地 Mac，2-3天）
目标：复现 A1 结论（手工特征 vs 深度学习）

```
对比: EEG特征 vs ERP特征 vs EEGNet vs EEGConformer
数据集: 同上（控制变量）
关注: F1 Score 差异是否与论文表2一致
```

### Phase 3（云服务器，1-2周）
目标：复现 A4 结论（patch embedding 对比）

```
实验: Multi-variate vs Uni-variate vs Whole-variate
数据集: 全部12个
硬件: 4×GPU（论文用 RTX A5000）
云推荐: AutoDL 租 4090 × 2
```

---

## M2 本地跑时的注意事项

```python
# 在代码中使用 MPS 加速（M2 的 GPU）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
data = data.to(device)

# M2 内存限制：建议 batch_size 从 32 开始（论文用 128）
# 如果OOM，调小 batch_size 并增大梯度累积步数
```

---

## 云服务器推荐配置（后期）

| 平台 | 推荐配置 | 预估费用 |
|------|---------|---------|
| AutoDL | RTX 4090 × 2 | ¥4-6/小时 |
| 阿里云 | A10 × 4 | ¥8-12/小时 |
| Colab Pro+ | A100 | $50/月 |

论文用了 4×RTX A5000，实际上 2×4090 完全可以复现主要实验。

---

## 常见问题

**Q: MNE 是什么？**
A: MNE（MNE-Python）是EEG/MEG信号处理的标准库，相当于EEG领域的sklearn。预处理pipeline全靠它。

**Q: 数据预处理需要多久？**
A: 每个数据集首次预处理大约1-4小时（主要是ICA去伪迹）。处理后保存为.npz文件，后续直接加载。

**Q: subject-independent 是什么意思？**
A: 训练集和测试集的受试者完全不重叠。比如127个受试者按6:2:2分，76人训练，25人验证，26人测试。这比subject-dependent难很多，也更接近实际应用。
