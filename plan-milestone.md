# ERP-Benchmark 复现计划与里程碑

更新日期：2026-05-01

本文档用于持续记录 ERP-Benchmark 复现进展。后续每推进一个阶段，优先在对应章节追加“实验设置、关键结果、发现的问题、代码改动、下一步计划”。

## 0. 总体目标

最终目标是复现论文《Benchmarking ERP Analysis: Manual Features, Deep Learning, and Foundation Models》的主要结论，而不只是跑通单个模型。

当前复现主线：

| 阶段 | 目标 | 当前状态 |
| --- | --- | --- |
| Phase 1 | 跑通 `CESCA-AODD + EEGNet` 完整 pipeline，并拿到可信 baseline | 已完成可信 baseline |
| Phase 2 | 复现 A1：手工特征 vs 深度学习 | 已完成 10 个数据集统一 `MPS` 对比，2 个数据集因单类问题跳过 |
| Phase 3 | 复现 A4：patch embedding 对比 | AutoDL 云端准备中 |

当前建议的推进顺序：

1. 固化 Phase 1 的可信 EEGNet baseline。
2. 完成 Phase 2，比较 `EEGFeatures / ERPFeatures / EEGNet / EEGConformer`。
3. 推进 Phase 3 的 `Multi-variate / Uni-variate / Whole-variate` patch embedding 对比，并迁移到 AutoDL GPU 环境执行。

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

目标：复现 patch embedding 对 ERP 建模影响的结论。Phase 2 已经完成 10 个数据集统一 `MPS` 对比，Phase 3 不再建议基于本地 Mac 长跑，应迁移到云端 GPU，并把环境、数据、脚本、结果回收都提前自动化，以减少按时计费的空转时间。

### 3.1 对比范围

| 对比项 | 说明 | 状态 |
| --- | --- | --- |
| `Multi-variate` | 多变量 patch embedding，现有脚本 `scripts/TestFormer/supervised/TestFormer/S-1-Multi-Variate.sh` | 待云端 sanity |
| `Uni-variate` | 单变量 patch embedding，现有脚本 `scripts/TestFormer/supervised/TestFormer/S-1-Uni-Variate.sh` | 待云端 sanity |
| `Whole-variate` | 全变量 patch embedding，现有脚本 `scripts/TestFormer/supervised/TestFormer/S-1-Whole-Variate.sh` | 待云端 sanity |

当前代码中已有 `run.py --patch_type`，可选值为 `multi-variate / uni-variate / whole-variate`；模型入口为 `models/TestFormer.py`。因此 Phase 3 的主要工作不是新增模型，而是做云端实验调度、失败恢复、日志汇总和成本控制。

### 3.2 AutoDL API 调研结论

官方文档确认 AutoDL 支持两类可用 API：

| 路线 | 能力 | 适用性 |
| --- | --- | --- |
| 容器实例 Pro API | 支持开发者 Token 鉴权、创建实例、查询实例状态、开机、关机、释放、保存镜像、获取镜像列表；创建实例时可指定 GPU 数量、规格 ID、镜像 UUID、CUDA 下限和开机后执行命令 | 推荐作为个人账号优先路线，适合“租一台实例 -> 跑完整实验 -> 关机/释放”的训练任务 |
| 弹性部署 API | 支持创建 `ReplicaSet / Job / Container`，可设置 GPU 型号、GPU 数量、CPU/内存/价格范围、镜像 UUID、启动命令，并可查询容器、停止容器、停止/删除部署、查询 GPU 库存 | 更适合批量 Job 化训练，但官方文档写明需要先认证企业；若账号不具备企业能力，则作为备选 |

关键限制与注意事项：

1. AutoDL API Host 为 `https://api.autodl.com`，Token 在控制台设置页获取。
2. 容器实例 Pro API 创建实例默认是按量计费，创建接口文档标注暂不支持通过 API 选择其他计费方式。
3. 容器实例 Pro API 的 `start_command` 只是在开机后执行，失败不会导致实例自动关机，因此训练脚本内部必须自己做失败退出、结果打包、主动关机或外部 watchdog。
4. 弹性部署 API 的自定义镜像需要先在 AutoDL 网页创建并保存，文档标注暂不支持从外部导入镜像。
5. 弹性部署 API 有 GPU 库存查询接口，但文档提醒库存默认按 1 张卡筛选；如果一个容器需要 2 卡，库存结果不保证可调度到同一台机器。
6. 计费按容器开机到关机的时间计算，不以是否实际调用 GPU 为准；因此所有下载、解压、依赖安装都应该尽量在镜像或数据盘预热阶段完成。

参考链接：

| 内容 | 链接 |
| --- | --- |
| 容器实例 Pro API | `https://www.autodl.com/docs/instance_pro_api/` |
| 弹性部署 API | `https://www.autodl.com/docs/esd_api_doc/` |
| 计费规则 | `https://www.autodl.com/docs/price/` |

### 3.3 推荐执行路线

Phase 3 建议分三轮，不直接一口气跑全量：

| 轮次 | 目标 | 数据集 | patch 类型 | 运行设置 | 放行条件 |
| --- | --- | --- | --- | --- | --- |
| Round 0：本地/CPU 脚本预检 | 验证脚本生成、参数、结果目录、跳过逻辑 | `CESCA-AODD` | 3 种 | `itr=1`, `train_epochs=1` | 三种 patch type 都能完整训练、验证、测试并写出 `results.txt` |
| Round 1：AutoDL 单卡 sanity | 验证 CUDA 环境、数据路径、显存、日志回收 | `CESCA-AODD` | 3 种 | `itr=1`, 短 epoch 或较小 patience | 三种 patch type 在 CUDA 上无 OOM、无 loader 错误，结果可自动回收 |
| Round 2：AutoDL 正式批量 | 复现 A4 主结论 | 优先 Phase 2 已完成的 10 个有效数据集；`PD-SIM / PD-ODD` 暂不进入正式结论 | 3 种 | 先 `itr=1`，稳定后再考虑 `itr=5` | 每个有效数据集都有三种 patch type 的 Test F1/AUROC/AUPRC，并能生成汇总表 |

不建议一开始就跑原脚本里的全量 `12 datasets × 3 patch types × itr=5 × 200 epochs`。按现有 Phase 2 经验，`PD-SIM / PD-ODD` 当前存在训练集单类问题，直接纳入会浪费云端时间，也会污染 Phase 3 对比口径。

### 3.4 租机前准备清单

租用 AutoDL 前先在本地完成：

1. 新增 Phase 3 调度脚本：从数据集列表和 patch type 生成单个实验命令，支持 `--dataset`、`--patch_type`、`--itr`、`--epochs`、`--patience`、`--batch_size`。
2. 新增数据集预检：复用 Phase 2 的单类标签检查，正式批量前自动跳过 `single_class_train_labels` 数据集。
3. 新增结果跳过逻辑：若目标 `results.txt` 已存在且包含 Test 指标，则默认不重复跑；允许 `--force` 覆盖。
4. 新增云端 bootstrap 脚本：完成 `git clone/pull`、虚拟环境创建、依赖安装、`python verify_env.py`、数据目录检查、1 epoch smoke test。
5. 新增结果打包脚本：训练结束后压缩 `results/TestFormer`、`checkpoints/TestFormer`、运行日志和环境信息。
6. 准备数据策略：优先把 `dataset/200Hz` 放在 AutoDL 数据盘或网盘缓存中；不把大数据下载放进正式计费窗口。
7. 准备环境镜像：首次手动创建 AutoDL 实例，安装依赖并保存私有镜像；后续 API 创建实例直接使用该 `image_uuid`。
8. 准备密钥与配置：本地保存 `AUTODL_TOKEN`、`AUTODL_IMAGE_UUID`、目标 GPU 规格、地区、预算上限；这些配置不提交到 git。

### 3.5 AutoDL 自动化设计

建议新增 `scripts/autodl/` 目录，职责分层如下：

| 脚本 | 职责 |
| --- | --- |
| `bootstrap_phase3.sh` | 云端实例启动后执行，负责进入项目、同步代码、校验 CUDA、校验数据、启动 Phase 3 runner |
| `run_phase3_matrix.py` | 读取数据集和 patch type 矩阵，串行或并行调度实验，支持断点续跑 |
| `collect_phase3_results.py` | 汇总 `results/TestFormer/.../results.txt`，输出 CSV/Markdown 表 |
| `autodl_pro_client.py` | 调用容器实例 Pro API：创建实例、查状态、查详情、关机、释放 |
| `autodl_watchdog.py` | 本地监控实例状态和最大运行时长，超过预算或训练结束后调用关机/释放 |

容器实例 Pro API 的首选流程：

1. 手动准备一次基础实例，安装依赖，确认 `torch.cuda.is_available()`，保存为私有镜像。
2. 本地脚本调用 `/api/v1/dev/instance/pro/create`，指定 `gpu_spec_uuid`、`req_gpu_amount`、`image_uuid`、`cuda_v_from` 和 `start_command`。
3. 轮询 `/api/v1/dev/instance/pro/status` 和 `/api/v1/dev/instance/pro/snapshot`，拿到 SSH 信息后确认训练开始。
4. 训练脚本完成后写出 `PHASE3_DONE` 标记并打包结果。
5. 本地 watchdog 检测完成标记或超过最大时长，调用 `/power_off`，确认关机后再 `/release`。

弹性部署 API 的备选流程：

1. 若账号完成企业认证，优先使用 `deployment_type=Job`，把每个 patch type 或数据集切成独立 Job。
2. 创建前调用 GPU 库存接口，筛选地区、CUDA、GPU 型号和价格范围。
3. 使用 `reuse_container=true` 减少重复创建容器耗时。
4. 通过容器查询接口回收 SSH、状态和价格信息；失败容器可设置调度黑名单后重试。

### 3.6 资源与成本策略

初始推荐配置：

| 项目 | 建议 |
| --- | --- |
| GPU | 先用单卡 `RTX 4090` 或 `4090-48G` 跑 sanity；确认显存后再考虑 2 卡并行 |
| 并行方式 | 优先“多进程多实验分卡”，不优先改模型 DDP；现有脚本只是设置 `CUDA_VISIBLE_DEVICES=0,1,2,3`，并不等于已经实现多卡训练 |
| batch size | 从原脚本 `128` 开始；若 OOM，降到 `64/32`，记录为云端修订设置 |
| 正式范围 | 先跑 Phase 2 已确认有效的 10 个数据集；`PD-SIM / PD-ODD` 另开数据标签修复任务 |
| 成本止损 | 每轮设置最大运行时长、最大失败次数、结果目录已存在跳过；训练完成必须自动关机 |

### 3.7 下一步

1. 本地先实现 Phase 3 runner、结果汇总和 AutoDL bootstrap，不立即租机。
2. 用 `CESCA-AODD × 3 patch types × 1 epoch` 做脚本级 smoke test。
3. 手动创建一次 AutoDL 实例，安装环境并保存镜像，记录 `image_uuid`。
4. 通过容器实例 Pro API 做一次短租 sanity，验证自动启动、训练、结果打包、关机/释放全链路。
5. Sanity 通过后再执行 10 个有效数据集的正式 Phase 3 批量实验。

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
