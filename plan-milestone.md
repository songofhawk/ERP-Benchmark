# ERP-Benchmark 复现计划与里程碑

更新日期：2026-04-07

本文档用于持续记录 ERP-Benchmark 复现进展。后续每推进一个阶段，优先在对应章节追加“实验设置、关键结果、发现的问题、代码改动、下一步计划”。

## 0. 总体目标

最终目标是复现论文《Benchmarking ERP Analysis: Manual Features, Deep Learning, and Foundation Models》的主要结论，而不只是跑通单个模型。

当前复现主线：

| 阶段 | 目标 | 当前状态 |
| --- | --- | --- |
| Phase 1 | 跑通 `CESCA-AODD + EEGNet` 完整 pipeline，并拿到可信 baseline | 已完成可信 baseline |
| Phase 2 | 复现 A1：手工特征 vs 深度学习 | 已完成 10 个数据集统一 `MPS` 对比，2 个数据集因单类问题跳过 |
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
| `EEGFeatures` | `scripts/EEGFeatures/supervised/EEGFeatures/S-1.sh` | 已完成 `CESCA-AODD` 长跑，并补跑 `MPS` |
| `ERPFeatures` | `scripts/ERPFeatures/supervised/ERPFeatures/S-1.sh` | 已完成 `CESCA-AODD` 长跑，并补跑 `MPS` |
| `EEGNet` | 当前 Phase 1 已有修正版 baseline | 已完成 `CESCA-AODD` 长跑，并补跑 `MPS` |
| `EEGConformer` | `scripts/EEGConformer/supervised/EEGConformer/S-1.sh` | 已完成 `CESCA-AODD` 长跑（MPS） |

### 2.1 当前实验设置

本轮 Phase 2 先在同一个数据集 `CESCA-AODD` 上控制变量，只比较方法差异，不扩大到多数据集。

| 项目 | 设置 |
| --- | --- |
| 数据集 | `CESCA-AODD` |
| 训练 / 测试 | `training_datasets=CESCA-AODD`, `testing_datasets=CESCA-AODD` |
| 类别不平衡处理 | 统一开启 `--use_class_weights` |
| 运行次数 | 当前先固定 `itr=1`，对应 seed 41 |
| 运行环境 | 当前四种方法都已有本机 `MPS` 长跑结果 |

### 2.2 已完成实验

| 方法 | 结果路径 | 关键 Test 结果 |
| --- | --- | --- |
| `EEGFeatures` | `results/EEGFeatures/supervised/EEGFeatures/S-CESCA-AODD-phase2-long-mps/results.txt` | F1 49.89%, AUROC 49.26% |
| `ERPFeatures` | `results/ERPFeatures/supervised/ERPFeatures/S-CESCA-AODD-phase2-long-mps/results.txt` | F1 51.77%, AUROC 55.30% |
| `EEGNet` | `results/EEGNet/supervised/EEGNet/S-CESCA-AODD-canonical-balanced-long-mps/results.txt` | F1 52.00%, AUROC 59.18% |
| `EEGConformer` | `results/EEGConformer/supervised/EEGConformer/S-CESCA-AODD-phase2-long-mps/results.txt` | F1 57.45%, AUROC 62.85% |

当前按统一 `MPS` 环境下的 `Test F1 / AUROC` 排序为：

1. `EEGConformer`
2. `EEGNet`
3. `ERPFeatures`
4. `EEGFeatures`

补跑 `MPS` 后的观察：

1. `EEGFeatures` 的 `MPS` 结果与原 CPU 长跑数值一致，说明其训练与评估在当前设置下基本稳定。
2. `ERPFeatures` 的 `MPS` 结果与原 CPU 长跑有轻微数值差异（Test F1 从 52.99% 变为 51.77%），但方法排序没有变化。
3. `EEGNet` 的 `MPS` 长跑结果低于此前 CPU baseline（Test F1 从 53.81% 变为 52.00%），说明当前实验在不同设备路径下仍可能存在一定数值波动。
4. 当前四种方法都已经有本机 `MPS` 结果，因此后续若强调公平对比，应优先引用 `-mps` 版本结果。

### 2.3 当前观察与结论

1. 在 `CESCA-AODD` 上，当前单数据集对比结果支持“深度学习方法整体优于手工特征方法”的趋势。
2. `EEGConformer` 当前是四种方法里最优，在统一 `MPS` 对比下仍明显领先。
3. `ERPFeatures` 明显优于 `EEGFeatures`，说明 ERP 任务中更贴近诱发成分的手工特征仍然有效。
4. 当前结论仍然只是 `CESCA-AODD + seed 41` 条件下的受控对比，还不能直接替代论文 A1 的全结论。

### 2.4 本阶段新增问题与修复

| 问题 | 影响 | 修复位置 |
| --- | --- | --- |
| `EEGConformer` 在 `MPS` 验证阶段报 `Placeholder storage has not been allocated on MPS device` | 训练能跑，但第一次验证就崩溃 | `exp/exp_supervised.py` |
| 本地 shell 默认 `python` 指向 `pyenv`，容易与项目 `.venv` 混用 | `verify_env.py` 与训练脚本可能落在不同 `torch` 环境 | `scripts/EEGNet/supervised/EEGNet/S-1-local-mac.sh` |
| Codex 沙箱内 `MPS` 设备不可见 | 在沙箱内运行时会自动回退到 CPU，误判为环境问题 | 启动命令需使用非沙箱方式 |

### 2.5 跨数据集总览

当前已完成统一 `MPS` 对比的数据集共有 10 个：

| 数据集 | 最优方法 | Test F1 | Test AUROC |
| --- | --- | ---: | ---: |
| `CESCA-AODD` | `EEGConformer` | 57.45% | 62.85% |
| `CESCA-VODD` | `EEGConformer` | 70.67% | 79.82% |
| `CESCA-FLANKER` | `EEGConformer` | 63.85% | 68.43% |
| `mTBI-ODD` | `EEGConformer` | 62.79% | 84.92% |
| `NSERP-MSIT` | `EEGNet` | 35.70% | 64.29% |
| `NSERP-ODD` | `EEGConformer` | 63.23% | 88.90% |
| `AOPD` | `EEGNet` | 75.08% | 87.06% |
| `ADHD-WMRI` | `EEGConformer` | 69.93% | 78.60% |
| `SCPD` | `EEGNet` | 74.28% | 83.26% |
| `RLPD` | `ERPFeatures` | 71.62% | 78.14% |

按“每个数据集谁是最优方法”统计：

| 方法 | 获胜数据集数 |
| --- | ---: |
| `EEGConformer` | 6 |
| `EEGNet` | 3 |
| `ERPFeatures` | 1 |
| `EEGFeatures` | 0 |

按 10 个数据集的平均 Test 指标统计：

| 方法 | 平均 Test F1 | 平均 Test AUROC |
| --- | ---: | ---: |
| `EEGConformer` | 62.19% | 77.03% |
| `EEGNet` | 61.17% | 75.41% |
| `ERPFeatures` | 57.97% | 70.87% |
| `EEGFeatures` | 51.56% | 62.48% |

### 2.6 跳过数据集

以下两个 disease detection 数据集在当前处理后标签与现有 loader 组合下，训练集只包含单一类别，因此自动跳过：

| 数据集 | 状态 | 原因 |
| --- | --- | --- |
| `PD-SIM` | skipped | `single_class_train_labels` |
| `PD-ODD` | skipped | `single_class_train_labels` |

### 2.7 当前阶段结论

1. 在当前 10 个已完成数据集上，`EEGConformer` 是最稳定的强方法，赢下了 6 个数据集。
2. `EEGNet` 是第二稳定的强基线，在 `AOPD`、`SCPD`、`NSERP-MSIT` 上取得最优结果。
3. `ERPFeatures` 虽然整体不如最强深度模型，但在 `RLPD` 上取得了最优结果，说明手工特征在部分 disease detection 任务上仍有竞争力。
4. `EEGFeatures` 在当前 10 个数据集里没有取得最优，但作为轻量手工特征基线仍可提供稳定对照。
5. 当前结果已经明显超出单数据集 sanity check，可以作为 Phase 2 的阶段性结论基础，但仍不是论文 A1 的严格最终复现。

### 2.8 下一步

1. 评估是否把当前 `CESCA-AODD` 结果扩展到更多 ERP 数据集，再对照论文 A1 结论。
2. 若继续 Phase 2，优先考虑增加重复随机种子或扩展到第二个 ERP 数据集，而不是直接跳到全量 12 数据集。
3. 若要分析设备影响，可单独整理 CPU vs MPS 的差异，而不要与方法对比混写。
4. 若要补完 12 个数据集，需要先确认 `PD-SIM / PD-ODD` 当前处理后标签列是否与 loader 假设一致，再决定是否修 loader 或重做标签映射。
5. 在 Phase 2 结论稳定后，再开始 Phase 3 的 patch embedding 对比。

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
