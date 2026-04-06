"""
ERP Benchmark 环境验证脚本
运行方式: python verify_env.py

验证内容:
1. 核心库是否安装
2. M2 MPS 加速是否可用
3. ERP 数据结构模拟
4. 一个最简单的 EEGNet 前向传播
"""

import sys

def check_imports():
    print("=" * 50)
    print("Step 1: 检查依赖库")
    print("=" * 50)
    
    packages = {
        "torch": "PyTorch (深度学习框架)",
        "mne": "MNE (EEG信号处理)",
        "numpy": "NumPy (数值计算)",
        "scipy": "SciPy (科学计算)",
        "sklearn": "Scikit-learn (机器学习工具)",
        "pandas": "Pandas (数据处理)",
        "matplotlib": "Matplotlib (可视化)",
    }
    
    all_ok = True
    for pkg, desc in packages.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {desc:<30} v{ver}")
        except ImportError:
            print(f"  ✗ {desc:<30} 未安装！")
            all_ok = False
    
    return all_ok


def check_device():
    import torch
    print("\n" + "=" * 50)
    print("Step 2: 检查计算设备")
    print("=" * 50)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  ✓ Apple M2 MPS 加速可用 (推荐使用)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  ✓ CUDA GPU 可用: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  ! 仅 CPU 可用 (速度较慢，但可以跑通)")
    
    print(f"  → 将使用设备: {device}")
    return device


def simulate_erp_data():
    import torch
    import numpy as np
    
    print("\n" + "=" * 50)
    print("Step 3: 模拟 ERP 数据结构")
    print("=" * 50)
    
    # 模拟 CESCA-AODD 数据集的一个batch
    # 论文Table 1: 38,151 trials, 26 channels, 200Hz, [-0.2, 0.8]秒
    SAMPLE_RATE = 200       # Hz
    EPOCH_START = -0.2      # 秒 (基线开始)
    EPOCH_END = 0.8         # 秒 (刺激后结束)
    N_TIMEPOINTS = int((EPOCH_END - EPOCH_START) * SAMPLE_RATE)  # = 200
    N_CHANNELS = 26
    BATCH_SIZE = 32
    N_CLASSES = 2           # 标准刺激 vs 目标刺激
    
    print(f"\n  数据集参数 (CESCA-AODD):")
    print(f"    采样率:    {SAMPLE_RATE} Hz")
    print(f"    时间窗:    [{EPOCH_START}, {EPOCH_END}] 秒")
    print(f"    时间点数:  {N_TIMEPOINTS} 个")
    print(f"    通道数:    {N_CHANNELS} 个")
    print(f"    类别数:    {N_CLASSES} (0=标准, 1=目标)")
    
    # 生成模拟数据 (实际数据从.npz文件读取)
    # 论文格式: (N_trials, T_timepoints, C_channels)
    X = torch.randn(BATCH_SIZE, N_TIMEPOINTS, N_CHANNELS)
    y = torch.randint(0, N_CLASSES, (BATCH_SIZE,))
    
    print(f"\n  模拟数据张量:")
    print(f"    X.shape = {list(X.shape)}")
    print(f"            = (batch, timepoints, channels)")
    print(f"    y.shape = {list(y.shape)}")
    print(f"    y 样本  = {y[:8].tolist()} ...")
    
    # ERP特征说明：基线期 vs 刺激期
    baseline_samples = int(0.2 * SAMPLE_RATE)  # 前40个点是基线
    evoked_samples = N_TIMEPOINTS - baseline_samples  # 后160个点是诱发电位
    print(f"\n  时间轴拆解:")
    print(f"    基线期: 前 {baseline_samples} 个时间点 (-0.2 ~ 0.0 秒) → 用于基线校正")
    print(f"    诱发期: 后 {evoked_samples} 个时间点 (0.0 ~ 0.8 秒) → 包含ERP成分")
    
    return X, y, N_CHANNELS, N_TIMEPOINTS


def run_minimal_eegnet(X, y, n_channels, n_timepoints, device):
    """
    EEGNet 最简实现 (用于验证环境)
    原始论文: Lawhern et al. 2018
    特点: 极其轻量 (~10K参数), 专为EEG设计
    """
    import torch
    import torch.nn as nn
    
    print("\n" + "=" * 50)
    print("Step 4: 运行最简 EEGNet 前向传播")
    print("=" * 50)
    
    class MinimalEEGNet(nn.Module):
        """
        EEGNet 核心结构:
        1. 时域卷积: 提取频率特征 (相当于带通滤波)
        2. 深度卷积: 空间滤波 (每个通道独立学习权重)
        3. 可分离卷积: 捕捉时域特征
        4. 分类头: 全连接层输出类别
        """
        def __init__(self, n_channels, n_timepoints, n_classes=2, 
                     F1=8, D=2, F2=16):
            super().__init__()
            
            # Block 1: 时域 + 空域卷积
            self.temporal_conv = nn.Sequential(
                # 时域卷积: kernel覆盖0.5秒 (SAMPLE_RATE//2 = 100)
                nn.Conv2d(1, F1, kernel_size=(1, n_timepoints//2), 
                         padding=(0, n_timepoints//4), bias=False),
                nn.BatchNorm2d(F1),
            )
            self.spatial_conv = nn.Sequential(
                # 深度卷积: 在通道维度做空间滤波
                nn.Conv2d(F1, F1*D, kernel_size=(n_channels, 1), 
                         groups=F1, bias=False),
                nn.BatchNorm2d(F1*D),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(0.25),
            )
            
            # Block 2: 可分离卷积
            self.separable_conv = nn.Sequential(
                nn.Conv2d(F2, F2, kernel_size=(1, 16), 
                         padding=(0, 8), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8)),
                nn.Dropout(0.25),
            )
            
            # 计算展平后的维度
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_timepoints)
                dummy = self.temporal_conv(dummy)
                dummy = self.spatial_conv(dummy)
                dummy = self.separable_conv(dummy)
                flat_dim = dummy.flatten(1).shape[1]
            
            # 分类头
            self.classifier = nn.Linear(flat_dim, n_classes)
        
        def forward(self, x):
            # 输入: (batch, timepoints, channels)
            # 调整为: (batch, 1, channels, timepoints) → Conv2d格式
            x = x.permute(0, 2, 1).unsqueeze(1)
            x = self.temporal_conv(x)
            x = self.spatial_conv(x)
            x = self.separable_conv(x)
            x = x.flatten(1)
            return self.classifier(x)
    
    # 实例化模型
    model = MinimalEEGNet(n_channels, n_timepoints, n_classes=2)
    model = model.to(device)
    X_dev = X.to(device)
    y_dev = y.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  模型参数量: {total_params:,} ({total_params/1000:.1f}K)")
    print("  (EEGNet 极其轻量, 这是它的设计优势)")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        logits = model(X_dev)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
    
    print(f"\n  前向传播结果:")
    print(f"    输入:   X.shape = {list(X.shape)}")
    print(f"    输出:   logits.shape = {list(logits.shape)}")
    print(f"    预测:   {preds[:8].cpu().tolist()} ...")
    print(f"    概率:   {probs[:3, 1].cpu().tolist()} ... (类别1的概率)")
    
    # 验证损失计算
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y_dev)
    print(f"    损失:   {loss.item():.4f} (初始约为 ln(2)≈0.693)")
    
    print(f"\n  ✓ EEGNet 前向传播成功！")
    return model


def print_next_steps():
    print("\n" + "=" * 50)
    print("Step 5: 接下来做什么")
    print("=" * 50)
    print("""
  1. 准备处理后的数据集:
     把 CESCA-AODD 放到 dataset/200Hz/CESCA-AODD/
     并确认存在 Feature/ 和 Label/ 两个目录

  2. 查看仓库运行说明:
     cat README.md

  3. 用本地 Mac 示例脚本跑第一个 EEGNet 实验:
     bash scripts/EEGNet/supervised/EEGNet/S-1-local-mac.sh

  4. 或直接调用统一入口脚本:
     python run.py --method EEGNet --task_name supervised --is_training 1 \
       --root_path ./dataset/200Hz/ --model_id S-CESCA-AODD-canonical-balanced-long \
       --model EEGNet --data MultiDatasets --training_datasets CESCA-AODD \
       --testing_datasets CESCA-AODD --batch_size 32 --use_class_weights \
       --des 'Canonical-Balanced-Long' --itr 1 --learning_rate 0.0001 \
       --train_epochs 30 --patience 5

  5. 之后再按同样方式切换到其他模型:
     例如把 --model EEGNet 改成 --model EEGConformer

  如遇问题, 可以把报错信息发给我分析！
    """)


if __name__ == "__main__":
    print("\nERP Benchmark 环境验证")
    print("论文: Benchmarking ERP Analysis (arXiv:2601.00573)\n")
    
    ok = check_imports()
    if not ok:
        print("\n请先安装缺失的库，再运行此脚本。")
        sys.exit(1)
    
    device = check_device()
    X, y, n_ch, n_t = simulate_erp_data()
    model = run_minimal_eegnet(X, y, n_ch, n_t, device)
    print_next_steps()
    
    print("\n" + "=" * 50)
    print("✓ 所有验证通过！环境已就绪。")
    print("=" * 50 + "\n")
