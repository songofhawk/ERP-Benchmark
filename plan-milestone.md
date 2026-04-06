# ERP-Benchmark 复现计划与里程碑

更新日期：2026-04-07

本文档用于持续记录 ERP-Benchmark 复现进展。后续每推进一个阶段，优先在对应章节追加“实验设置、关键结果、发现的问题、代码改动、下一步计划”。

## 0. 总体目标

最终目标是复现论文《Benchmarking ERP Analysis: Manual Features, Deep Learning, and Foundation Models》的主要结论，而不只是跑通单个模型。

当前复现主线：

| 阶段 | 目标 | 当前状态 |
| --- | --- | --- |
| Phase 1 | 跑通 `CESCA-AODD + EEGNet` 完整 pipeline，并拿到可信 baseline | 已完成可信 baseline |
| Phase 2 | 复现 A1：手工特征 vs 深度学习 | 待开始 |
| Phase 3 | 复现 A4：patch embedding 对比 | 待开始 |

当前建议的推进顺序：

1. 固化 Phase 1 的可信 EEGNet baseline。
2. 开始 Phase 2，比较 `EEGFeatures / ERPFeatures / EEGNet / EEGConformer`。
3. 在 Phase 2 结果稳定后，再推进 Phase 3 的 `Multi-variate / Uni-variate / Whole-variate` patch embedding 对比。

## 1. Phase 1：CESCA-AODD + EEGNet

### 1.1 目标

Phase 1 的定位是 pipeline sanity check，不是论文最终结论本身。

目标数据和模型：

| 项目 | 设置 |
| --- | --- |
| 数据集 | `CESCA-AODD` |
| 数据路径 | `dataset/200Hz/CESCA-AODD` |
| 模型 | `EEGNet` |
| 关键指标 | `Accuracy / F1 / AUROC / AUPRC` |

### 1.2 数据准备

已下载作者提供的处理后数据：

| 项目 | 当前状态 |
| --- | --- |
| `Feature` 文件数 | 50 |
| `Label` 文件数 | 50 |
| 数据大小 | 约 664 MB |
| 样本形状示例 | `feature_001.npy: (335, 200, 26)` |
| 标签形状示例 | `label_001.npy: (335, 4)` |

数据来源：

| 类型 | 链接 |
| --- | --- |
| 处理后数据入口 | `https://drive.google.com/drive/folders/1pVUmPlsQN9j5HD5YJSeiDrBAUAKBCQA5?usp=drive_link` |
| `CESCA-AODD` 原始数据 | `https://openneuro.org/datasets/ds006018/versions/1.2.2` |

### 1.3 已发现并修复的问题

| 问题 | 影响 | 修复位置 |
| --- | --- | --- |
| `feature` 和 `label` 通过 `zip(os.listdir(...))` 配对 | macOS 下目录顺序不稳定，导致 subject 错配，出现大量假 mismatch | `data_provider/uea.py` |
| `VAL/TEST` 评估阶段 `drop_last=True` | 验证和测试会丢掉最后一个 batch | `data_provider/data_factory.py` |
| NumPy 2.x 不支持 `np.Inf` | 训练在 EarlyStopping 初始化处崩溃 | `utils/tools.py` |
| Mac 本地设备选择只考虑 CUDA | Apple Silicon / CPU 环境不顺 | `run.py`, `exp/exp_basic.py` |
| 类别分布约 `79:21` | 无权重训练容易塌缩成全预测多数类 | `exp/exp_supervised.py` 新增 `--use_class_weights` |
| 原 `EEGNet` 实现偏自定义 `TemporalSpatialConv` | 表征能力弱，且和标准 EEGNet 偏离较大 | `models/EEGNet.py` 改为标准 EEGNet-style 结构 |

### 1.4 关键诊断结果

数据对齐修复前，训练日志出现大量 mismatch，例如 `Subject 47 data and label length mismatch`。排查后确认真实文件本身没有长度错配，错配来自 loader 文件配对逻辑。

真实数据切分在 seed 41 下：

| Split | 样本数 | Subject 数 |
| --- | ---: | ---: |
| Train | 10028 | 30 |
| Val | 3343 | 10 |
| Test | 3344 | 10 |

类别分布约为：

| Split | 0 类 | 1 类 | 多数类比例 |
| --- | ---: | ---: | ---: |
| Train | 7934 | 2094 | 79.12% |
| Val | 2644 | 699 | 79.09% |
| Test | 2647 | 697 | 79.16% |

无类别权重时，模型基本全预测 0 类。曾验证 test 混淆矩阵为：

```text
[[2633    0]
 [ 695    0]]
```

### 1.5 已完成实验

| 实验 ID | 说明 | 结果路径 | 关键 Test 结果 |
| --- | --- | --- | --- |
| `S-CESCA-AODD-local` | 初始本地跑通版本，后续判断为不可信 baseline | `results/EEGNet/supervised/EEGNet/S-CESCA-AODD-local/results.txt` | F1 44.18%, AUROC 51.72% |
| `S-CESCA-AODD-canonical-balanced` | 标准 EEGNet + class weights，8 epoch 诊断短跑 | `results/EEGNet/supervised/EEGNet/S-CESCA-AODD-canonical-balanced/results.txt` | F1 50.75%, AUROC 55.83% |
| `S-CESCA-AODD-canonical-balanced-long` | 标准 EEGNet + class weights，30 epoch / patience 5 长跑 | `results/EEGNet/supervised/EEGNet/S-CESCA-AODD-canonical-balanced-long/results.txt` | F1 53.81%, AUROC 61.01% |

当前可信 Phase 1 baseline：

| 指标 | Validation | Test |
| --- | ---: | ---: |
| Accuracy | 63.48% | 60.74% |
| Precision | 57.34% | 55.35% |
| Recall | 60.55% | 57.81% |
| F1 | 56.41% | 53.81% |
| AUROC | 64.31% | 61.01% |
| AUPRC | 58.85% | 56.40% |

对应 checkpoint：

```text
checkpoints/EEGNet/supervised/EEGNet/S-CESCA-AODD-canonical-balanced-long/nh8_el2_dm512_df2048_seed41/checkpoint.pth
```

### 1.6 Phase 1 结论

Phase 1 已经完成，当前 baseline 比最初结果更可信。

核心判断：

1. 初始 `Accuracy ~79%` 是多数类基线现象，不应该被当作模型表现好。
2. 数据配对 bug 和评估 drop-last bug 会显著污染结果，已经修复。
3. `CESCA-AODD` 存在明显类别不平衡，后续实验建议默认开启 `--use_class_weights` 或进一步尝试 sampler。
4. 标准 EEGNet 结构明显优于原自定义轻量结构，当前结果可以作为 Phase 2 的 EEGNet baseline 参考。

## 2. Phase 2：A1 结论复现计划

目标：复现论文中关于手工特征和深度学习方法对比的结论。

建议先在同一个数据集 `CESCA-AODD` 上控制变量：

| 方法 | 脚本参考 | 状态 |
| --- | --- | --- |
| `EEGFeatures` | `scripts/EEGFeatures/supervised/EEGFeatures/S-1.sh` | 待跑 |
| `ERPFeatures` | `scripts/ERPFeatures/supervised/ERPFeatures/S-1.sh` | 待跑 |
| `EEGNet` | 当前 Phase 1 已有修正版 baseline | 已有 baseline |
| `EEGConformer` | `scripts/EEGConformer/supervised/EEGConformer/S-1.sh` | 待跑 |

Phase 2 待办：

1. 确认 `EEGFeatures / ERPFeatures / EEGConformer` 是否受 `load_data_by_ids()` 修复影响并能正常跑。
2. 为所有 supervised 方法统一使用修复后的 `VAL/TEST drop_last=False`。
3. 评估是否所有方法都应使用 `--use_class_weights`，或是否只作为“不平衡修正版”单独报告。
4. 整理四种方法在 `CESCA-AODD` 上的 F1 / AUROC 排序，和论文 A1 结论对照。

## 3. Phase 3：A4 结论复现计划

目标：复现 patch embedding 对 ERP 建模影响的结论。

建议在 Phase 2 完成后推进：

| 对比项 | 说明 | 状态 |
| --- | --- | --- |
| `Multi-variate` | 多变量 patch embedding | 待跑 |
| `Uni-variate` | 单变量 patch embedding | 待跑 |
| `Whole-variate` | 全变量 patch embedding | 待跑 |

Phase 3 备注：

1. 全量 12 数据集会明显增加耗时。
2. 本地 Mac 更适合做小规模 sanity check。
3. 大规模复现建议迁移到 GPU 云服务器。

## 4. 后续更新模板

每次新增一章或更新一个阶段时，建议使用以下模板：

```markdown
## N. 标题

### N.1 目标

### N.2 实验设置

### N.3 结果

### N.4 问题与修复

### N.5 结论

### N.6 下一步
```

## 5. Phase 1 归档范围

Phase 1 当前建议归档的关键改动：

| 类型 | 路径 |
| --- | --- |
| 数据加载修复 | `data_provider/uea.py` |
| 评估 DataLoader 修复 | `data_provider/data_factory.py` |
| 设备选择修复 | `run.py`, `exp/exp_basic.py` |
| 类别权重训练 | `run.py`, `exp/exp_supervised.py` |
| EEGNet 标准化实现 | `models/EEGNet.py` |
| NumPy 2.x 兼容 | `utils/tools.py` |
| 本地运行脚本 | `scripts/EEGNet/supervised/EEGNet/S-1-local-mac.sh` |

进入 Phase 2 前，应尽量避免把新的实验改动混进 Phase 1 的修复范围。如果需要继续修改训练策略，建议单独开新的 commit 或在本文档新增 Phase 2 章节。
