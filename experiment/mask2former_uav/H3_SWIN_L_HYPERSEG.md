# H3：Swin-L 骨干 + 仓库现有 HyperSeg 解码路径

状态：实验设计与训练入口已落地；H3 已在 2026-09-19 完成 160000 iter。最佳验证 mIoU 77.66%（137200 iter），最终 iter mIoU 77.45%；使用最佳 checkpoint 的独立 test mIoU 为 78.73%。H0 配对重跑仍待实现和执行。

## 1. 目的与对照

检验将 MiT-B3 替换为 Swin-L 后，仓库现有 HyperSeg 解码路径能否获益。这里的“现有 decoder”指 `hyperseg_uav/model.py` 中编码器之后的完整单图路径，不是 `fewshot_hyperseg` 的 support/query 模型，也不只是末端两层卷积。

| 编号 | 骨干 | 解码路径 | 状态 |
| --- | --- | --- | --- |
| H0 | MiT-B3 | 仓库 HyperSeg | 原实现已有；需按本文协议重跑配对基线 |
| H1 | MiT-B3 | Mask2Former | 已完成，验证 mIoU 76.51% |
| H2 | Swin-L | Mask2Former | 计划中 |
| H3 | Swin-L + 通道投影 | 仓库 HyperSeg | 本文新增设计 |

主要比较 H3 与配对重跑的 H0。H3 与 H2 可比较同一骨干下的两套解码及损失方案，但须先统一数据、预算和评估协议。Mask2Former 与 HyperSeg 损失和输出约定不同，不能把二者差异全部归因于解码器结构。

## 2. 架构

```text
RGB [B,3,H,W]，ImageNet 标准化
  → Swin-L，四级步长 [4,8,16,32]
  → 特征通道 [192,384,768,1536]
  → 各级独立 1×1 Conv（bias=True，无额外激活或归一化）
  → 对齐到 [64,128,320,512]
  → SceneCondition(context_dim=128)
  → DynamicModulation(scale=0.15)
  → DynamicFusion(out_channels=64)
  → DynamicLowRankAdapter(rank=8, scale=0.1)
  → 原有两层 3×3 Conv + BatchNorm + GELU，始终为 64 通道
  → 原有 segmentation head（9 通道）与 boundary head（1 通道）
  → 双线性插值回输入尺寸
```

Swin-L 使用 patch4/window12/384、ImageNet-22K 预训练版本：stage depths `[2,2,18,2]`，heads `[6,12,24,48]`，drop-path 0.3，输出四个 stage 的 NCHW 特征。384 是预训练尺寸，不要求 UAV 输入缩放至 384；训练使用 512 裁剪，注意力窗口所需 padding 由骨干实现处理。

投影是新增的可学习接口，随机初始化并参与训练。H0 使用 Identity 接口，后续模块的结构、通道和初始化规则与 H3 相同。比较结论应表述为“替换骨干并增加必要通道适配后的收益”，不是绝对零额外参数的骨干消融。

不直接把 HyperSeg 的 `widths` 改成 `[192,384,768,1536]`：现有代码以 `widths[0]` 决定融合、低秩模块和末端 decoder 的宽度，这会同时将 decoder 从 64 加宽到 192，改变实验问题。

## 3. 模型实现与初始化来源

沿用当前服务器 OpenMMLab 环境，H3 骨干采用 MMSegmentation 的 `SwinTransformer`，解码路径复用仓库 PyTorch 模块。不要直接加载为 Hugging Face Swin，两个实现的 state_dict 命名和格式不同。

- 官方配置：<https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py>
- Swin-L 骨干权重：<https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth>
- 配对 H0 保持 Hugging Face `nvidia/mit-b3` 编码器来源，与原 HyperSeg 实现一致。

H3 仅加载 Swin-L 骨干权重，通道投影和所有 HyperSeg 下游模块重新初始化；不加载 ADE20K 完整 Mask2Former 权重，也不继承 H0 已训练的 decoder。记录权重来源、文件 SHA-256、加载时缺失/多余键以及骨干配置；不允许骨干大面积未加载而继续正式训练。

该对比包含模型规模及预训练数据差异，不能声称单独证明 Swin 架构优于 MiT。若随后实现 H2，应使用同一 Swin-L 骨干权重，以形成同骨干对照。

## 4. H0/H3 配对训练协议

| 项目 | 两组统一设置 |
| --- | --- |
| 数据 | 固定 train 5598 / val 699 / test 699，复用 `runs/splits` |
| 标签 | 保留原始 0..8；9 通道输出；`ignore_index=0`，有效类别 1..8 |
| 训练裁剪 | 512×512 |
| 数据增强 | 复用 `UAVDataset`：离散随机缩放、场景/稀有类裁剪、水平/垂直翻转、亮度/对比度/颜色扰动 |
| 输入标准化 | RGB [0,1] 后减 mean `[0.485,0.456,0.406]`，除 std `[0.229,0.224,0.225]`；只执行一次 |
| 损失 | 原 `hyperseg_loss`：0.55 focal CE + 0.30 Dice + 0.025 rare + 0.10 boundary BCE |
| 优化器 | AdamW，LR 1e-4，骨干 LR 1e-5，weight decay 0.01，betas=(0.9,0.999) |
| 参数分组 | 预训练骨干使用低 LR；H3 新增投影及 HyperSeg 所有下游模块使用完整 LR |
| 学习率计划 | PolyLR，power=0.9，160000 次 optimizer update 衰减至 0 |
| 梯度裁剪 | 全模型 global norm 1.0，在 optimizer step 前执行 |
| 样本预算 | 有效 batch 2，160000 次 optimizer update，共 320000 个训练样本视图，约 57.2 个等效 epoch |
| 精度 | 两组均 FP32；资源不足时先启用骨干 gradient checkpointing |
| 验证 | 每 2800 次 optimizer update，结束时额外验证 |
| 随机种子 | 首轮 3407；扩展复现使用 3408、3409，两组配对运行 |
| 模型选择 | 验证集全局 mIoU 最佳；独立 test 不参与选模或调参 |

增强保持图像/掩码空间对齐，图像用双线性插值、掩码用最近邻插值。当前 `UAVDataset` 只输出 [0,1]，未做上述标准化，因此该步骤必须在新入口中显式实现并用于训练、验证、测试和导出。

不要直接使用旧 H0 成绩作配对结论：原训练入口使用 768 缩放验证、固定学习率和 batch mIoU 平均。本文的 H0 是重新训练的受控基线。另外，H1 使用不同增强与优化细节，现有 76.51% 只作参考，不是严格受控对比。

如 Swin-L 无法使用 micro-batch 2，则 H0/H3 都采用 micro-batch 1、累积 2 次；每次反传将 loss 除以 2，每次 optimizer update 后才更新 scheduler。总计 320000 次 micro-step，仍为 160000 次 optimizer update。BatchNorm 依赖实际 micro-batch，梯度累积并不等价于 batch 2，因此配对基线也必须采用同样的 micro-batch。

正式训练前固定两组的 micro-batch、累积数和精度，并记录实际设置；不在运行中静默改变预算。首轮一个种子仅作初步判断，多种子报告均值和标准差。

## 5. 评估与交付

验证和独立测试均使用原始分辨率 whole-image 推理、batch 1、无 TTA。保留 9 通道 argmax；真实标签为 0 的像素完全排除，真实有效而预测为 0 的像素仍算错误。

累计整个数据集的 9×9 混淆矩阵后，对类别 1..8 计算 IoU 和 mIoU；禁止平均每张图或每个 batch 的 mIoU。全局 union 为 0 的类别记为 NaN 并排除，报告实际参与平均的类别。同步报告各类 IoU、有效像素 aAcc/mAcc，重点比较 Barren 和 Background。

除准确率外，记录总/可训练参数量、峰值训练显存、训练耗时、同一 GPU 上原始尺寸 batch-1 推理延迟。延迟测试先 warm-up，计时前后同步 CUDA，不包含磁盘读写。

checkpoint 必须包含完整模型权重，以及骨干类型/配置、通道适配配置、decoder 配置、标准化、类别及 ignore 约定；恢复训练另存 optimizer、scheduler、随机数/采样状态、更新步数和最佳验证指标。推理应从本地配置重建模型后加载完整 checkpoint，不依赖重新下载骨干初始化权重。

建议产物命名：

```text
runs/mask2former_uav/h0_mit_b3_hyperseg_512_seed3407/
runs/mask2former_uav/h3_swin_l_hyperseg_512_seed3407/
  train.log
  config.json
  best.pt
  last.pt
  val_metrics.json
  test_metrics.json
```

训练在服务器后台运行，日志写入对应工作目录。训练结束后，将日志和指标下载到本实验目录的 `logs/`、`results/`；权重保留服务器，不提交仓库。无标签比赛预测需通过现有 submission checker：同名、单通道 L、1024×1024、像素 0..8。

## 6. 实现与启动前验收

1. `hyperseg_uav/swin_l_model.py` 和 `train_h3_hyperseg.py` 提供实验专用模型/训练入口，复用现有 decoder 和损失；`run.py train-h3` 提供统一启动包装，不改变旧入口行为。
2. 验证 512×512 和 1024×1024 前向的四级特征、投影及最终输出形状；确认 mask 未发生双线性缩放或标签重映射。
3. 在服务器完成 20 次更新的 smoke test：有限 loss/梯度，确认骨干、投影及 decoder 参与反传，检查预训练加载日志。
4. 检查 checkpoint 保存/恢复、全局混淆矩阵及 Ignore 规则，使用包含“有效像素预测为 0”的小例子验证计分。
5. 测定训练与原图验证显存，确定配对 micro-batch 后，再后台启动 H0/H3 正式运行。

当前 `run.py --model mask2former` 与 `tools/train_hyperseg.py` 不负责 H3；请使用
`run.py train-h3`（底层为 `train_h3_hyperseg.py`）的训练入口，使用
`--eval-checkpoint` 做验证/测试，使用 `infer_h3_hyperseg.py` 做无标签导出。
正式训练仍须先通过上述 smoke test 和显存验收。
