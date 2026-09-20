# 手动支持集：HyperSeg-UAV v2 的 0/1/2/5/10-shot 实验

入口为 `experiment/loveda_fewshot/run_manual.py`。使用 `tools/model_1.py` 的 v2 结构，默认源权重为 `models/hyperseg_resume_best.pt`。旧 `train.py` / `run_sweep.py` 保留作每域 K 张的历史实验；本入口只复用其中的数据增强、损失等公共函数。

## 当前支持集与协议

已导出目录：`runs/loveda_manual/export_20260912_101105_509451`。

| 每类别 K | 去重后的训练图像数 | 固定评估图像数 |
|---:|---:|---:|
| 0 | 0 | 1599 |
| 1 | 7 | 1599 |
| 2 | 14 | 1599 |
| 5 | 35 | 1599 |
| 10 | 70 | 1599 |

每类取用户选择列表的前 K 张，跨类取并集，使用完整语义 mask；不是每类独立训练一个二分类器，也不是按目标实例计数。所有 K 排除全部 70 张支持图片，共用相同评估图片。清单、嵌套顺序、交集、文件存在性在运行前检查。支持集不按 seed 重新抽样。

这是固定支持集的跨数据集少样本监督微调；不是 episodic 元学习。因为支持图来自 LoveDA Val，结果应标注为 Val 内部分割，不是官方完整 Val 指标。手工选择有选择偏差，三个 seed 的标准差仅反映训练随机性。

## 模型迁移与 0-shot

严格检查源 checkpoint 的 `model_config.version=v2`、9 个源通道以及所有参数的名称和尺寸；支持已知的 Transformers 5 到 4 encoder 参数名转换，其他不匹配直接报错。8 通道目标头复制源通道 0..7，删除 Vehicle 通道 8。通道 6 使用源 Vegetation 近似初始化 Forest，Background 的定义也可能有数据集差异。这是语义映射的跨数据集 0-shot 基线，不能称为开放词汇零样本分割。

0-shot 不读支持 mask 来估计权重、不建优化器、不训练，只评估迁移后的模型；没有随机初始化的目标类别，因此只运行一次。所有正 K、mode、seed 均重新从同一个源 checkpoint 初始化，不接着小 K 的模型训练。

## 默认训练与评估

- 主实验 `adapter`：冻结 encoder，训练 scene、modulation、fusion、low_rank、decoder、**boundary_refine** 和两个预测头。`head` 只训练两个预测头；`full` 训练全部参数。
- 所有 mode 均冻结 BatchNorm 的 running statistics；可训练模块的 BN 仿射参数仍按 mode 更新。冻结模块保持 eval 状态。
- 每个正 K 运行 seed 3407/3408/3409；每次 2000 个训练 batch，batch size 2，512 裁剪。有放回图像均匀采样，沿用随机缩放、翻转和颜色增强。
- AdamW，学习率 `1e-4`，weight decay `0.01`；full 的 encoder 学习率为其 0.1 倍；多项式学习率衰减，CUDA 默认 AMP（支持时使用 BF16，否则 FP16），梯度裁剪 1.0。FP16 梯度溢出会降低缩放系数并重试同一 batch，不计为有效更新。类别权重仅由本次支持 mask 计算。
- 最终步模型作为主结果，**不根据固定评估集挑 epoch、不回滚最佳模型**。因此评估集只用于最终报告，不用于模型选择。预算应预先固定；若要调参，应另设开发集。
- 评估为 1024 整图，无 TTA；Ignore 0/255 不计入指标，预测仅在 1..7 中 argmax。统计全评估集混淆矩阵、7 类 IoU/mIoU 和像素准确率。

默认总计 13 次任务（1 个 0-shot + 4 个 K × 3 个训练 seed），在服务器按顺序执行。可增添 `head/full` 作微调范围消融。

## 运行

本地只检查清单和图像/mask 文件存在性，无需 PyTorch，不会启动训练；此检查不验证 GPU、模型数值前向或骨干缓存：

```bash
python experiment/loveda_fewshot/run_manual.py --manifest-dir runs/loveda_manual/export_20260912_101105_509451 --check-only
python -m unittest experiment.loveda_fewshot.test_manual_protocol
```

服务器需项目 `requirements.txt` 依赖、LoveDA Val、上述完整导出目录、v2 模型脚本和源 checkpoint。同时需要本地可加载的 Hugging Face mit-b3 配置及骨干权重：优先 `models/nvidia--mit-b3/`，其次 Hugging Face 缓存；也可用 `--backbone-path /path/to/mit-b3` 指定包含配置和权重的目录。加载骨干时不联网。

从服务器项目根目录运行完整主实验：

```bash
python experiment/loveda_fewshot/run_manual.py \
  --data-root LoveDA \
  --manifest-dir runs/loveda_manual/export_20260912_101105_509451 \
  --init-checkpoint models/hyperseg_resume_best.pt \
  --output-root runs/loveda_manual_v2
```

先运行单 seed 完整 5 组实验，可加 `--seeds 3407`。添加微调范围消融可加 `--modes head adapter full`。不同训练预算、分辨率或超参数用新的 `--output-root`；显存不足可降 `--batch-size`，但正式比较应对所有 K 使用相同预算和配置。

默认需要 CUDA。本地只做轻量检查；`--device cpu` 仅供明确选择的调试使用，不会自动退回 CPU 完整训练。

## 输出与重跑

每组目录如 `adapter_5shot_seed3407/`，包含 `run_config.json`、`selection.json`、`history.jsonl`、`final.pt`、`summary.json`。0-shot 目录 `zero_0shot/` 只保存配置、清单和结果，不重复保存模型。

根目录 `results.csv` 包含每次运行的 mIoU、各类 IoU、实际支持图数；`aggregate.json` 按 mode/K 汇总 mIoU 均值和样本标准差。数值范围为 0..1。每完成一组更新一次汇总。

为防止混用结果，记录清单和源 checkpoint 的 SHA-256，并将配置、关键代码内容纳入运行签名。默认拒绝覆盖已有组目录。完整批次重新执行时加 `--skip-completed` 可跳过签名相同且已有 `summary.json` 的组。没有实现训练中途续训；中断组需使用新输出根目录重新运行所需 K/seed，已有目录保留。汇总仅覆盖本次命令请求的任务，完整汇总请用完整任务参数和 `--skip-completed` 重跑。

`final.pt` 保存的是 **8 通道 v2 模型**；旧 `tools/test_hyperseg.py` 使用旧架构，不能直接加载它。当前入口负责训练并立即完成最终评估。

## 单种子的更新范围与训练预算对照

新增入口 `run_ablation.py`，默认只使用 seed 3407，对 1-shot 和 2-shot 各执行以下三个新设置，共六次训练：

| 设置 | 可训练部分 | 更新步数 |
|---|---|---:|
| adapter_200 | 原 adapter 范围，含 decoder 和 boundary_refine | 200 |
| semantic_head_2000 | 仅语义分类头 weight/bias | 2000 |
| semantic_head_200 | 仅语义分类头 weight/bias | 200 |

`semantic-head` 是新模式，不同于旧 `head`（后者还训练边界预测头）。当前 64 通道特征、8 类输出的语义头只有 520 个参数。新运行的 `run_config.json` 记录可训练参数总数、总参数数和可训练参数名称，用于核对冻结范围。

每次仍从原始 v2 UAV checkpoint 初始化，复用原先的嵌套支持集和 1599 张固定评估图。初始学习率 1e-4、batch size 2、512 裁剪、1024 评估、BF16 和损失组成均保持原设置。边界头冻结时，边界损失不产生可训练参数的梯度，不改变语义损失的系数。学习率按各自的总步数完成多项式衰减，200 步结果不是旧 2000 步训练轨迹的前缀。评估只在最终步执行。

不重跑 0-shot、adapter 2000 步或 5/10-shot；比较时使用已有 **seed 3407** 的结果，而不是旧三种子均值。已有基线 mIoU：0-shot 53.3395%，1-shot adapter-2000 49.8105%，2-shot adapter-2000 51.6253%。单种子结果仅用于此次探索性对照。

先查看计划（不启动 GPU、不写文件）：

```bash
python experiment/loveda_fewshot/run_ablation.py --manifest-dir runs/loveda_manual/export_20260912_101105_509451 --dry-run
```

服务器项目根目录启动：

```bash
python -u experiment/loveda_fewshot/run_ablation.py \
  --data-root /root/autodl-tmp/LoveDA \
  --manifest-dir runs/loveda_manual/export_20260912_101105_509451 \
  --seed 3407 --output-root runs/loveda_ablation_v2
```

所有子任务的标准输出和报错统一写入 `runs/loveda_ablation_v2/ablation_<时间戳>.log`，同时输出到终端。每个设置独立保存模型、历史和指标，根目录生成 `ablation_results.csv`，不与旧结果混写。任一子任务失败后停止后续设置。`--skip-completed` 沿用主入口的签名检查；中断组不支持中途续训。

本地轻量验证：`python -m unittest experiment.loveda_fewshot.test_ablation`。
