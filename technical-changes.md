# 技术变更记录

更新日期：2026-04-07

本文档记录复现过程中对代码和本地环境适配做过的技术变更。`plan-milestone.md` 记录复现进度和实验结论；本文档记录代码层面的“改了什么、为什么改、如何验证、潜在影响”。

## 0. 变更总览

| 类别 | 文件 | 目的 |
| --- | --- | --- |
| 数据加载 | `data_provider/uea.py` | 按 subject ID 显式匹配 `feature/label`，避免目录顺序导致错配 |
| DataLoader 评估逻辑 | `data_provider/data_factory.py` | `VAL/TEST` 不再丢最后一个 batch |
| 设备选择 | `run.py`, `exp/exp_basic.py` | 支持 `cuda / mps / cpu`，适配本地 Mac |
| 类别不平衡 | `run.py`, `exp/exp_supervised.py` | 新增 `--use_class_weights`，可启用 inverse-frequency weighted CE |
| 模型实现 | `models/EEGNet.py` | 将原自定义 `TemporalSpatialConv` 版本替换为标准 EEGNet-style 结构 |
| NumPy 兼容 | `utils/tools.py` | `np.Inf` 替换为 `np.inf`，兼容 NumPy 2.x |
| 本地脚本 | `scripts/EEGNet/supervised/EEGNet/S-1-local-mac.sh` | 增加本地 Phase 1 运行入口 |
| Git 忽略 | `.gitignore` | 忽略 `paper/` 与 `.venv/` 等本地大文件或环境目录 |

## 1. 数据加载修复：按 Subject ID 配对

### 问题

原始 `load_data_by_ids()` 使用：

```python
for feature_filename, label_filename in zip(os.listdir(data_path), os.listdir(label_path)):
```

这依赖两个目录的 `os.listdir()` 返回顺序一致。macOS 上这个顺序不稳定，因此会出现 `feature_045.npy` 被错误配到 `label_041.npy` 这类问题。训练日志里出现了大量看似是数据问题的 mismatch，例如：

```text
Subject 47 data and label length mismatch
```

进一步检查发现真实 `feature_047.npy` 与 `label_047.npy` 的长度是匹配的，mismatch 来自 loader 错配。

### 修改

位置：`data_provider/uea.py`

现在先把 `Feature/` 和 `Label/` 目录分别解析成 subject ID -> filename 的映射，然后按 subject ID 显式配对加载。

关键逻辑：

```python
feature_file_map = build_subject_file_map(data_path)
label_file_map = build_subject_file_map(label_path)

for sub_id in sorted(ids):
    sub_feature_path = os.path.join(data_path, feature_file_map[sub_id])
    sub_label_path = os.path.join(label_path, label_file_map[sub_id])
```

### 验证

修复后，`CESCA-AODD` 在 seed 41 下加载结果为：

| Split | 修复后样本数 |
| --- | ---: |
| Train | 10028 |
| Val | 3343 |
| Test | 3344 |

不再出现假 mismatch。

### 影响

这是关键修复。它会影响所有通过 `load_data_by_ids()` 加载的处理后数据集。后续 Phase 2/3 实验都应该基于这个修复版本运行。

## 2. 评估 DataLoader 修复：VAL/TEST 不再 Drop Last

### 问题

原代码判断：

```python
if flag == 'test':
```

但项目实际传入的是大写 `TRAIN / VAL / TEST`。因此 `VAL/TEST` 会落入训练分支，并被强制 `drop_last=True`。这会导致验证集和测试集最后一个 batch 被丢弃。

### 修改

位置：`data_provider/data_factory.py`

现在先统一：

```python
flag = flag.upper()
```

然后：

```python
if flag in ('VAL', 'TEST'):
    shuffle_flag = False
    drop_last = False
else:
    shuffle_flag = True
    drop_last = True
```

### 验证

在 `batch_size=32` 下：

| Split | 样本数 | DataLoader batch 数 |
| --- | ---: | ---: |
| Train | 10028 | 313 |
| Val | 3343 | 105 |
| Test | 3344 | 105 |

`VAL/TEST` batch 数符合不丢尾 batch 的预期。

### 影响

评估指标现在覆盖完整验证集和测试集。历史结果中使用旧逻辑得到的指标应视为不完全可比。

## 3. 设备选择修复：CUDA / MPS / CPU

### 问题

原代码只判断 CUDA：

```python
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
```

在 Mac 本地会退回 CPU，且后续仍有无条件调用 `torch.cuda.*` 的风险。

### 修改

位置：

| 文件 | 修改 |
| --- | --- |
| `run.py` | 设置 `args.device_type` 为 `cuda / mps / cpu` |
| `exp/exp_basic.py` | 根据 `args.device_type` 获取设备 |
| `run.py` | 仅在 CUDA 下调用 `torch.cuda.manual_seed_all()` 和 `torch.cuda.empty_cache()` |

### 影响

本地 Mac 能更稳定运行。当前环境实际仍显示 `device_type='cpu'`，因为当前 `.venv` 的 torch 未检测到 MPS 可用，但代码路径已兼容 MPS。

## 4. 类别权重：`--use_class_weights`

### 问题

`CESCA-AODD` 类别分布约为 `79:21`：

| Split | 0 类 | 1 类 | 多数类比例 |
| --- | ---: | ---: | ---: |
| Train | 7934 | 2094 | 79.12% |
| Val | 2644 | 699 | 79.09% |
| Test | 2647 | 697 | 79.16% |

无类别权重时，模型基本塌缩成全预测 0 类。曾验证 test 混淆矩阵为：

```text
[[2633    0]
 [ 695    0]]
```

### 修改

位置：

| 文件 | 修改 |
| --- | --- |
| `run.py` | 新增 CLI 参数 `--use_class_weights` |
| `exp/exp_supervised.py` | 根据训练集标签频率计算 class weights |
| `exp/exp_supervised.py` | 启用 `nn.CrossEntropyLoss(weight=...)` |

权重计算：

```python
class_counts = np.bincount(train_data.y[:, 0].astype(int), minlength=self.args.num_class)
total_samples = class_counts.sum()
self.class_weights = torch.tensor(
    total_samples / (self.args.num_class * np.maximum(class_counts, 1)),
    dtype=torch.float32
)
```

### 影响

启用 `--use_class_weights` 后，模型不再全预测多数类。Phase 1 最终可信 baseline 使用了该选项。

后续 Phase 2 需要决定是否所有方法统一启用该选项，或者分别报告“原始训练”和“不平衡修正版”。

## 5. EEGNet 实现标准化

### 问题

原 `models/EEGNet.py` 实际使用 `TemporalSpatialConv`，不是标准 EEGNet-style 结构。排查时发现该实现结合类别不平衡后学习能力较弱，AUROC 长期贴近 0.5。

### 修改

位置：`models/EEGNet.py`

替换为标准 EEGNet-style 结构：

1. Temporal convolution。
2. Depthwise spatial convolution across EEG channels。
3. Separable temporal convolution。
4. Linear classifier。

当前参数量：

```text
1714
```

### 验证

`canonical EEGNet + --use_class_weights` 长跑结果：

| 指标 | Validation | Test |
| --- | ---: | ---: |
| Accuracy | 63.48% | 60.74% |
| Precision | 57.34% | 55.35% |
| Recall | 60.55% | 57.81% |
| F1 | 56.41% | 53.81% |
| AUROC | 64.31% | 61.01% |
| AUPRC | 58.85% | 56.40% |

结果路径：

```text
results/EEGNet/supervised/EEGNet/S-CESCA-AODD-canonical-balanced-long/results.txt
```

Checkpoint：

```text
checkpoints/EEGNet/supervised/EEGNet/S-CESCA-AODD-canonical-balanced-long/nh8_el2_dm512_df2048_seed41/checkpoint.pth
```

### 影响

当前 `EEGNet` baseline 与原仓库默认实现已经不同。进入 Phase 2 前，需要在报告中明确标注这是“修正后的 canonical EEGNet”。如果目标是严格复现论文原始代码结果，也可以保留一个分支或备份原实现进行对照。

## 6. NumPy 2.x 兼容

### 问题

当前环境使用 NumPy 2.x，`np.Inf` 已移除。训练会在 `EarlyStopping` 初始化时崩溃。

### 修改

位置：`utils/tools.py`

```python
self.val_loss_min = np.inf
```

### 影响

仅兼容性修复，不改变算法逻辑。

## 7. 本地运行脚本

### 新增文件

```text
scripts/EEGNet/supervised/EEGNet/S-1-local-mac.sh
```

用途：

1. 检查 `dataset/200Hz/CESCA-AODD/Feature` 和 `Label` 是否存在。
2. 默认使用 `.venv/bin/python`。
3. 使用当前 Phase 1 可信 baseline 参数运行 `CESCA-AODD + canonical EEGNet + --use_class_weights`。
4. 默认设置 `MPLCONFIGDIR=/tmp/matplotlib` 与 `XDG_CACHE_HOME=/tmp/.cache`，减少本地缓存权限问题。

注意：该脚本会写入 `S-CESCA-AODD-canonical-balanced-long` 对应的结果目录。若需要保留已有结果，请先修改 `--model_id`。

## 8. `.gitignore` 更新

新增忽略项：

```text
paper/
.venv/
```

已有忽略项包括：

```text
checkpoints/
results/
dataset/
```

目的：避免把本地环境、数据、结果、论文缓存等大文件误提交。

## 9. 当前推荐命令

当前 Phase 1 可信 baseline 对应命令：

```bash
MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp/.cache .venv/bin/python -u run.py \
  --method EEGNet \
  --task_name supervised \
  --is_training 1 \
  --root_path ./dataset/200Hz/ \
  --model_id S-CESCA-AODD-canonical-balanced-long \
  --model EEGNet \
  --data MultiDatasets \
  --training_datasets CESCA-AODD \
  --testing_datasets CESCA-AODD \
  --batch_size 32 \
  --use_class_weights \
  --des 'Canonical-Balanced-Long' \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --patience 5
```

## 10. 后续技术待办

| 优先级 | 待办 | 说明 |
| --- | --- | --- |
| 高 | 决定是否保留原 EEGNet 实现备份 | 严格复现论文原代码 vs 修正 baseline 需要说明 |
| 中 | 尝试 `WeightedRandomSampler` | 对比 class weights 是否更稳 |
| 中 | 为 Phase 2 方法统一检查类别权重与评估逻辑 | 保证对比公平 |
