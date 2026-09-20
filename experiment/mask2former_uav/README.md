# UAV Mask2Former experiment

This experiment evaluates Mask2Former + MiT-B3 (H1) against the repository's
existing HyperSegUAV architecture (H0). The standard MMSeg SegFormer decoder
config is an optional framework control, not H0. Mask2Former + Swin-L (H2)
is the competition-baseline comparison planned separately.
It uses the sibling `../mmsegmentation` checkout and does not modify that
repository. Comparisons require a matched data split and evaluation protocol.

H3（Swin-L 骨干 + 仓库现有 HyperSeg 解码路径）的新增实验设计见
[H3_SWIN_L_HYPERSEG.md](H3_SWIN_L_HYPERSEG.md)。该方案以通道投影保留
现有 decoder，并规定配对重跑 H0 的协议；H3 已完成 160000 iter，最佳验证 mIoU 为 **77.66%（137200 iter）**，最终 iter 的 mIoU 为 **77.45%**。
训练入口为 `train_h3_hyperseg.py`，使用服务器上的 MMSegmentation
Swin 实现；它支持断点恢复和独立 test/val 评估。

### H3 server2 启动记录（2026-09-18）

- 16:43:23（北京时间）后台启动，主训练 PID `2858`，启动脚本 PID `2852`。
- 工作目录：`/root/autodl-tmp/work_dirs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32`。
- 命令脚本：[launch_h3_server2.sh](launch_h3_server2.sh)。160000 optimizer updates，batch 2，累积 1，512 裁剪，FP32，seed 3407，启用 Swin gradient checkpointing。
- 数据：`/root/autodl-tmp/data/low_altitude_2026/train/{images,masks}`；固定划分 5598/699/699，服务器与本地划分 SHA-256 一致，所有掩码标签范围检查通过。
- Swin-L 骨干权重 SHA-256：`dc532f0ad8f98b71e0c88e8d673512b1526af9cba2cf0ab71130478423ffb3ff`。官方分类预训练文件不含四级输出 norm；这些层使用默认初始化，其余骨干参数通过加载检查。
- 启动前通过 20 次预训练更新、全部 699 张原图验证、checkpoint 保存与恢复、Ignore 指标检查。测试模型未用于正式训练。
- 启动检查至 350 iter，约 0.208 秒/update，峰值分配显存约 4.01 GB；速度不含后续完整验证开销，不代表最终耗时。
- [短程验证日志](logs/h3_smoke_pretrained.log)及[正式训练日志快照](logs/h3_swin_l_hyperseg_160k_seed3407_fp32/train.log)已下载。
- [sync_h3_server2_logs.py](sync_h3_server2_logs.py)每 5 分钟回收日志与配置，检测到服务器 `exit_code` 后保存最终日志并退出；依赖本地回收进程和网络持续可用，可重新运行该脚本恢复回收。服务器训练独立于本地连接运行。
- 最终 checkpoint 保留在服务器工作目录：`best.pt` 对应 137200 iter，`last.pt` 对应 160000 iter。

### H3 test 集评估（2026-09-19）

使用 `best.pt`（137200 iter）在固定 `test.txt` 的 699 张标注图片上做原始分辨率整图推理，得到 **mIoU 78.73%**。逐类 IoU 为：Background 71.11%、Building 85.29%、Road 81.70%、Water 90.10%、Barren 57.02%、Vegetation 89.01%、Agricultural 78.41%、Vehicle 77.19%。

- 指标文件：[best_test_metrics.json](logs/h3_swin_l_hyperseg_160k_seed3407_fp32/best_test_metrics.json)
- 评估日志：[test_eval.log](logs/h3_swin_l_hyperseg_160k_seed3407_fp32/test_eval.log)
- 该 test 结果只用于最终报告，不参与训练、选模或调参。

### H3 无标注比赛测试集导出（2026-09-19）

使用同一个 `best.pt` 对 `/root/autodl-tmp/data/low_altitude_2026/images` 的 **500 张**无标注图片完成滑窗推理。提交检查通过：500 个预测文件与输入文件名完全对应，均为单通道 PNG，尺寸为 1024×1024，像素值范围为 0..8。该目录没有真值标签，因此不能计算 mIoU。

- 本地预测目录：[unlabeled_test_predictions](results/h3_swin_l_hyperseg_160k_seed3407_unlabeled/unlabeled_test_predictions)
- 本地归档：[h3_unlabeled_test_predictions.tar.gz](results/h3_swin_l_hyperseg_160k_seed3407_unlabeled/h3_unlabeled_test_predictions.tar.gz)
- 归档 SHA-256：`950b495378f6fbea112b7c26e43b97b18535eb028746fc7588376f8f0f108ade`


也可以通过统一包装器启动（`--dry-run` 可先检查命令）：

```bash
python experiment/mask2former_uav/run.py train-h3 \
  --backbone-checkpoint /path/to/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth
```

正式长任务可在服务器后台运行：

```bash
mkdir -p runs/mask2former_uav/h3_swin_l_hyperseg_512_seed3407
nohup python -u experiment/mask2former_uav/run.py train-h3 \
  --backbone-checkpoint /path/to/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth \
  > runs/mask2former_uav/h3_swin_l_hyperseg_512_seed3407/train.log 2>&1 < /dev/null &
```

```bash
PYTHONPATH=../mmsegmentation:$PYTHONPATH python -u \
  experiment/mask2former_uav/train_h3_hyperseg.py \
  --data-root dataset \
  --backbone-checkpoint /path/to/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth \
  --work-dir runs/mask2former_uav/h3_swin_l_hyperseg_512_seed3407
```

评估已保存的 checkpoint：

```bash
PYTHONPATH=../mmsegmentation:$PYTHONPATH python \
  experiment/mask2former_uav/train_h3_hyperseg.py \
  --eval-checkpoint runs/mask2former_uav/h3_swin_l_hyperseg_512_seed3407/best.pt \
  --eval-split test --data-root dataset
```

对无标签图片导出预测：

```bash
PYTHONPATH=../mmsegmentation:$PYTHONPATH python \
  experiment/mask2former_uav/infer_h3_hyperseg.py \
  --checkpoint runs/mask2former_uav/h3_swin_l_hyperseg_512_seed3407/best.pt \
  --input dataset/low_altitude_2026/images \
  --output runs/mask2former_uav/h3_swin_l_hyperseg_512_seed3407/test_predictions
```

## H1 实验结果（2026-09-18）

H1 已在 server2 完成全部 160000 次迭代及最终验证，日志结束时间为
2026-09-18 00:30:02。最佳验证集 mIoU 为 **76.51%（148400 iter）**，
最终模型为 76.39%。以下均为固定 699 张验证集上的结果；本次记录尚不包含
独立测试集评估或比赛提交成绩。

### 运行配置

| 项目 | 本次设置 |
| --- | --- |
| 模型 | MiT-B3 预训练骨干 + Mask2Former pixel decoder / transformer decoder |
| 数据划分 | train 5598 / val 699 / test 699，固定划分 |
| 标签 | 原始 0 为 Ignore；原始 1..8 参与 mIoU，Background 是有效类 |
| 训练输入 | 随机缩放及 512×512 裁剪、水平/垂直翻转、颜色增强 |
| 验证方式 | 原始分辨率整图推理（whole），无 TTA |
| Batch / 累积 / 精度 | 2 / 1 / FP32 |
| 随机种子 | 3407 |
| 训练预算 | 160000 iter，约 57.2 个等效 epoch |
| 验证频率 | 每 2800 iter，训练结束时额外验证 |
| 优化器 | AdamW，初始 LR 1e-4，骨干 LR 倍率 0.1，weight decay 0.05 |
| 学习率与裁剪 | PolyLR（power=0.9，衰减至 0），梯度裁剪 max_norm=0.01 |
| 硬件与环境 | RTX 4090；PyTorch 2.3.0+cu121、MMCV 2.1.0、MMEngine 0.10.7、MMDetection 3.3.0、MMSegmentation 1.2.2 |

### 汇总与逐类指标

指标单位均为百分比。模型选择依据验证集 mIoU。

| Checkpoint | Iter | mIoU | aAcc | mAcc |
| --- | ---: | ---: | ---: | ---: |
| 最佳 | 148400 | 76.51 | 88.99 | 86.47 |
| 最终 | 160000 | 76.39 | 88.97 | 86.38 |

| 类别（原始 ID） | 最佳模型 IoU | 最终模型 IoU |
| --- | ---: | ---: |
| Background（1） | 70.50 | 70.38 |
| Building（2） | 84.88 | 85.02 |
| Road（3） | 80.81 | 80.76 |
| Water（4） | 85.05 | 84.82 |
| Barren（5） | 43.90 | 43.44 |
| Vegetation（6） | 88.75 | 88.71 |
| Agricultural（7） | 77.33 | 77.43 |
| Vehicle（8） | 80.86 | 80.53 |

最佳验证指标的后期演进：

| Iter | 当时最佳 mIoU |
| ---: | ---: |
| 81200 | 75.23 |
| 109200 | 76.07 |
| 123200 | 76.31 |
| 140000 | 76.46 |
| 148400 | 76.51 |

### 结论与后续评估

- FP32 正式训练完整结束，日志未发现 NaN、OOM 或匹配异常。此前 AMP
  启动失败不属于本次 FP32 成绩，具体限制见下方训练说明。
- 后期增益明显减小：109200 到最佳点提升 0.44 个百分点，140000 之后仅
  提升 0.05 个百分点。最终结果较最佳低 0.12 个百分点，验证曲线趋于稳定，
  未见持续大幅退化；仅凭这些验证指标不能严格排除过拟合。
- Barren（43.90 IoU）是主要短板，Background（70.50）次之。后续可通过
  混淆矩阵及预测可视化检查错误来源，当前日志尚不能确定具体混淆关系。
- 后续独立测试应使用 `best_mIoU_iter_148400.pth`。H0 需要按相同划分与
  原始分辨率评估协议对照；当前结果不足以证明 H1 优于仓库现有模型或 H2。

### 日志与模型位置

- 完整训练日志：[h1_mit_b3_160k_seed3407_fp32.train.log](logs/h1_mit_b3_160k_seed3407_fp32.train.log)
- server2 工作目录：`/root/autodl-tmp/work_dirs/mask2former_uav/h1_mit_b3_160k_seed3407_fp32`
- 最佳权重：该目录下 `best_mIoU_iter_148400.pth`
- 最终权重：该目录下 `iter_160000.pth`

本地已归档训练日志；权重仍位于服务器。以上结论来自该次单随机种子运行。

### 无标签测试集导出（2026-09-18）

使用 `best_mIoU_iter_148400.pth` 对
`/root/autodl-tmp/data/low_altitude_2026/images` 的 500 张比赛测试图片完成
整图推理。服务器端 `tools/check_submission.py` 校验通过：输出文件名与输入
完全一致，均为 1024×1024 单通道 PNG，类别值位于 `0..8`。该测试集没有
标注，因此本次导出没有 mIoU 等质量指标。

- 本地预测目录：[test_predictions](results/h1_mit_b3_160k_seed3407_fp32_test/test_predictions)
- 推理日志：[test.log](results/h1_mit_b3_160k_seed3407_fp32_test/test.log)
- 下载归档：[h1_test_predictions_20260918.tar.gz](results/h1_mit_b3_160k_seed3407_fp32_test/h1_test_predictions_20260918.tar.gz)
- 归档 SHA-256：`abe7513aff498948c09bf7c9c2217b5d2f094464edb8c4e4ea968f7d37df6f18`

## Label contract

Raw masks contain labels `0..8`, where `0` is Ignore.  MMSegmentation sees:

```text
raw 0    -> 255 (ignore)
raw 1..8 -> 0..7 (eight learned classes)
```

The submission config uses MMSegmentation's `reduce_zero_label` metadata to
add one back while saving, so exported PNG files contain `1..8`.  They are
then checked by the project's `tools/check_submission.py`.

## Environment

Use a separate server environment containing a CUDA-compatible PyTorch build
and these repository-compatible packages:

```text
mmsegmentation 1.2.2 (the sibling checkout, installed editable)
mmcv >= 2.0.0rc4, < 2.2.0
mmengine >= 0.5.0, < 1.0.0
mmdet >= 3.0.0rc4
```

Mask2Former needs MMDetection even though this is a semantic segmentation
experiment.  The default MiT-B3 initializer is the MMSegmentation checkpoint
URL.  For an offline server, download that file first and pass
`--backbone-checkpoint /path/to/mit_b3.pth`.

## Prepare and audit data

The expected labelled layout is:

```text
dataset/train/images/*.png
dataset/train/masks/*.png
runs/splits/{train,val,test}.txt
```

Extract the existing fixed splits without touching the dataset:

```bash
python experiment/mask2former_uav/run.py prepare-splits
```

After the dataset download completes, validate filenames, disjoint splits and
all mask IDs:

```bash
python experiment/mask2former_uav/run.py check --scan-masks
```

The fixed archive currently contains 5598 train, 699 validation and 699 test
IDs.  Split entries must be stems such as `0834`, not `0834.png`.

## Train

Run the primary experiment:

```bash
python experiment/mask2former_uav/run.py train \
  --model mask2former \
  --work-dir runs/mask2former_uav/mask2former_mit_b3_512
```

Run the framework control:

```bash
python experiment/mask2former_uav/run.py train \
  --model segformer \
  --work-dir runs/mask2former_uav/segformer_mit_b3_512
```

Both defaults use crop 512, batch 2, FP32, seed 3407 and 160k iterations.
At batch 2, this is about 57 passes over the 5598-image training split.  A
short server smoke test can be generated with:

```bash
python experiment/mask2former_uav/run.py train \
  --max-iters 20 --val-interval 20 \
  --work-dir runs/mask2former_uav/smoke
```

If batch 2 is out of memory, use batch 1 with two-step accumulation.  To keep
both the number of optimizer updates and viewed samples equivalent, double
the runner iterations and scheduler length through the wrapper:

```bash
python experiment/mask2former_uav/run.py train \
  --batch-size 1 --accumulative-counts 2 --max-iters 320000 \
  --val-interval 5600
```

AMP is intentionally disabled by default. In the server2 environment
(PyTorch 2.3, MMDetection 3.3), random training batches can make the
float16 Hungarian matching cost non-finite and raise `cost matrix is
infeasible`. Pass `--amp` only after separately validating a stabilized
mixed-precision matching implementation.

Add `--dry-run` to any train/eval/export command to print the exact command
and resolved paths without importing the OpenMMLab runtime.

## Evaluate and export

Evaluation uses the fixed labelled `test.txt` split at native resolution with
whole-image inference:

```bash
python experiment/mask2former_uav/run.py eval \
  --model mask2former \
  --checkpoint runs/mask2former_uav/mask2former_mit_b3_512/best_mIoU_iter_*.pth
```

Export the unlabelled 1024x1024 test directory:

```bash
python experiment/mask2former_uav/run_test.py \
  --checkpoint /path/to/best_mIoU_iter_148400.pth
```

`run_test.py` auto-detects the local
`dataset/low_altitude_2026/images` directory and the server layout
`../data/low_altitude_2026/images`. It also searches the completed H1 work
directory for `best_mIoU_iter_*.pth`; explicit `--input` and `--checkpoint`
always take priority. Use `--dry-run` to inspect resolved paths and the MMSeg
command without starting inference. A typical background server run is:

```bash
nohup .venv-mask2former/bin/python -u \
  experiment/mask2former_uav/run_test.py \
  > runs/mask2former_uav/h1_test.log 2>&1 < /dev/null &
```

Predictions default to
`runs/mask2former_uav/h1_mit_b3_160k_seed3407_fp32/test_predictions`.
The wrapper runs the existing submission checker after export, requiring 500
single-channel 1024x1024 PNGs with the original filenames and IDs in `0..8`.
Validation and
test selection deliberately use native-resolution whole-image inference
rather than the legacy HyperSeg 768x768 resize, so old and new mIoU values are
not directly comparable until the old model is re-evaluated with the same
protocol. MMSegmentation 1.2.2 slide inference is not used here because its
generic segmentor updates the crop `img_shape` but leaves the original
`pad_shape`; Mask2FormerHead prioritizes `pad_shape`, producing an incorrectly
sized crop prediction.

## Files

- `mask2former_mit_b3_512.py`: primary model and labelled train/val/test config.
- `segformer_mit_b3_512.py`: controlled standard SegFormer decoder.
- `*_submission.py`: unlabelled export variants.
- `run.py`: split preparation, data audit and reproducible command wrapper.
- `run_test.py`: H1 test-set path/checkpoint discovery and prediction export.
