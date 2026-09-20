# LoveDA few-shot 实验

该实验用 `models/hyperseg_b3_best.pt` 初始化 HyperSeg-UAV，在 LoveDA 上进行少样本监督微调。脚本只依赖官方 LoveDA 的目录结构，不包含数据下载和服务器部署。

**新的手动每类别 0/1/2/5/10-shot 实验已实现：**使用 `run_manual.py`、v2 权重 `models/hyperseg_resume_best.pt` 和固定 1599 张评估集，详见 [MANUAL_V2.md](MANUAL_V2.md)。下文的 `train.py` / `run_sweep.py` 仍描述旧版每域 K 张协议。

## GUI 手动挑选 Val 支持集（每类别 K 张）

从项目根目录、在有桌面的 Python 环境运行（只需 CPU，无需 PyTorch 或模型权重）：

```bash
python -m pip install numpy Pillow
python tools/select_loveda_gui.py --data-root LoveDA
```

Windows 官方 Python 安装器通常自带 tkinter（需勾选 Tcl/Tk）；Ubuntu 系统 Python 缺少 tkinter 时可安装 `python3-tk`。WSL 需要可用的 WSLg/X 显示服务，也可以直接在 Windows PowerShell 中运行。`--data-root` 可指向 LoveDA 根目录或 `LoveDA/Val`，兼容目录名大小写，要求同时有 Urban 和 Rural 的 `images_png` / `masks_png`。

默认选择全部 7 类，每类选 10 张。若“目标”不包括背景：

```bash
python tools/select_loveda_gui.py --classes 2 3 4 5 6 7 --output runs/loveda_manual_no_background
```

- 顶部切换目标类别、Urban/Rural、目标占整图的百分比区间，支持占比升序/降序和文件名排序。默认只列出包含当前目标的图片。
- 中间显示 RGB 与目标 mask 叠加，可调整透明度、切换原图或全部类别叠加。目标颜色对应右侧图例。左右方向键切图，空格加入当前类别；输入框内快捷键不触发。
- 图下同时显示目标像素数、占整图比例、占有效区域比例、Ignore 比例。统计基于原始 mask，`0` 和 `255` 均忽略。显示时图像双线性缩放、mask 最近邻缩放，不改变统计。
- 右侧显示已选列表；点击可回看，上移/下移调整顺序，移除后可替换图片。同一图片可被多个目标类别选中。
- 每次增删/排序自动保存至 `runs/loveda_manual/selection.json`，重启自动恢复；自定义位置用 `--output`。恢复时目标类别必须与保存时一致。GUI 不修改或复制数据集。

每个目标选满 10 张后，点击导出会创建新的 `export_<时间戳>/`，包含 `0shot.json`、`1shot.json`、`2shot.json`、`5shot.json`、`10shot.json` 和 `evaluation.json`，不会覆盖之前的导出。

这里采用的是**每类别 K 张图像**：各类顺序列表取前 K 张，组成嵌套支持集；`samples` 为跨类别去重后的图片并集，键为 `urban/文件名.png` 或 `rural/文件名.png`，因此实际图像数可能少于“类别数 × K”。这些是完整图像/完整语义 mask 的支持集，不是按实例计数或仅保留目标类别的二值标注；同一图片还可能额外包含其他类别。`0shot.json` 的支持集为空。

所有 K 共用同一个评估清单：从 Val 排除 **10-shot 全部已选图片的并集**，并记录在每个清单的 `evaluation_samples` 和单独的 `evaluation.json` 中。这样 0/1/2/5/10-shot 的评估图片一致。该方案属于 Val 内部划分，不能称为官方完整 Val 指标。

**与下文旧训练脚本的区别：**`train.py` 定义为“每域 K 张”，不会使用 `evaluation_samples`，也不支持 0-shot。请使用 [新的 v2 实验入口](MANUAL_V2.md) 接入 GUI 清单和固定评估集，不要直接套用下文旧批量命令。

轻量检查（无需显示器）：

```bash
python tools/test_select_loveda_gui.py
```

## 实验协议

- LoveDA 标签为 `0=no-data`、`1..7=background/building/road/water/barren/forest/agriculture`。模型改为输出 8 个通道，评估时忽略 0，对 1..7 计算 mIoU。
- K-shot 定义为每个域选 K 张带完整像素标注的训练图像。默认同时选 Urban 和 Rural，因此一次 5-shot 实验共有 10 张训练图像。
- 样本选择固定 seed，并用贪心策略优先覆盖当前欠缺的类别。每次运行把文件名和类别覆盖写入 `selection.json`。
- 训练用随机尺度、随机裁剪、翻转和轻量颜色增强；每轮通过有放回采样固定执行 `steps-per-epoch` 个 step，避免极小数据集每轮只有一两个梯度更新。
- 类别权重仅由选中的 few-shot 训练掩码估计。验证使用完整 LoveDA Val，模型选择指标为 7 类 mIoU。
- 如果暂时只有 Val，可同时传入 `--train-split Val --val-split Val`。脚本会从 Val 选择 few-shot 支持集，并自动从评估集排除这些图片，避免训练/验证重叠；该结果属于 Val 内部划分，不能作为官方 Val 指标。

对比三种微调范围：

| mode | 可训练部分 | 用途 |
|---|---|---|
| `head` | 语义头、边界头 | 最小参数基线 |
| `adapter` | 场景条件、动态调制/融合、低秩适配器、decoder 和 heads | 默认主实验 |
| `full` | 全模型 | 判断更多样本时全量微调是否获益 |

不同 mode 应复用同一份 `selection.json`。正式报告建议使用 K=`1,5,10,20`、三个 seed，报告 mIoU 的均值和标准差。

## 数据目录

脚本兼容目录名大小写，预期结构为：

```text
LoveDA/
  Train/
    Urban/images_png/*.png
    Urban/masks_png/*.png
    Rural/images_png/*.png
    Rural/masks_png/*.png
  Val/
    Urban/images_png/*.png
    Urban/masks_png/*.png
    Rural/images_png/*.png
    Rural/masks_png/*.png
```

## 单次实验

从项目根目录运行：

```bash
python experiment/loveda_fewshot/train.py \
  --data-root /path/to/LoveDA \
  --init-checkpoint models/hyperseg_b3_best.pt \
  --shots 5 \
  --mode adapter \
  --seed 3407 \
  --amp
```

只有 Val 时可运行：

```bash
python experiment/loveda_fewshot/train.py \
  --data-root /path/to/LoveDA \
  --train-split Val --val-split Val \
  --shots 5 --mode adapter --seed 3407 --amp
```

默认输出到 `runs/loveda_fewshot/adapter_5shot_seed3407/`：

- `selection.json`：few-shot 样本清单及类别覆盖；
- `run_config.json`：参数、类别像素统计和预训练权重迁移报告；
- `history.jsonl`：逐 epoch 损失与验证指标；
- `best.pt`：验证 mIoU 最佳的 checkpoint；
- `summary.json`：该次实验的最佳结果。

要让多个方法严格使用同一批样本，可传入首次运行生成的清单：

```bash
python experiment/loveda_fewshot/train.py \
  --data-root /path/to/LoveDA \
  --shots 5 --mode full --seed 3407 \
  --manifest runs/loveda_fewshot/adapter_5shot_seed3407/selection.json
```

## 批量对比

```bash
python experiment/loveda_fewshot/run_sweep.py \
  --data-root /path/to/LoveDA \
  --shots 1 5 10 20 \
  --seeds 3407 3408 3409 \
  --modes head adapter full \
  -- --epochs 50 --crop-size 512 --batch-size 2 --amp
```

批量脚本会自动让同一 K/seed 的不同 mode 复用样本，并生成 `sweep_results.csv` 和包含均值、标准差的 `sweep_summary.json`。

## 运行注意

- 初始化 HyperSeg 模型时会优先读取项目内的 `models/nvidia--mit-b3/`，不存在时再读取 Hugging Face 缓存；模型参数随后由项目 checkpoint 覆盖。
- `--eval-size 1024` 保持 LoveDA 官方分辨率。显存不足时可先设为 512 做调试，正式对比应为所有实验保持同一设置。
- 默认 `--restore-best`：每次验证 mIoU 没有刷新最佳值时，下一轮训练前恢复最佳 checkpoint 的模型、优化器、学习率调度器和 AMP 状态。使用 `--no-restore-best` 可关闭回滚。
- 默认主实验是 `adapter`。如果 few-shot 过拟合，可减少 `--steps-per-epoch` 或 epoch；所有方法必须使用相同训练预算。

## 分类头初始化对照

`--head-init random` 为默认基线，整个语义头保留随机初始化。
`--head-init mapped` 要求原 UAV 9 通道 checkpoint，复制通道 1、2、3、4、5、7 的 weight/bias 到 LoveDA 对应通道；0 和 6 保留随机初始化，原 Vehicle 通道 8 丢弃。分类头仍参与微调。Background 的标注范围可能不同，复制仅用于初始化，并不保证两套数据语义完全一致。
迁移报告中的 `skipped`/`missing` 记录整体张量加载情况；后续逐通道复制另记在 `head_mapping_target_to_source`，随机通道记在 `head_random_channels`。

server2 的固定对照使用原始 `models/hyperseg_b3_best.pt` 和基线 `selection.json`，保持 10 张支持图、1659 张评估图及全部训练超参数一致。开启 GPU 后，从服务器项目目录执行：

```bash
bash experiment/loveda_fewshot/start_mapped_control.sh
```

脚本检查 GPU，通过 `nohup` 后台运行，日志为 `runs/loveda_fewshot_valsplit/adapter_5shot_seed3407_mapped.log`，结果目录为同名无 `.log` 目录。目录已存在时拒绝启动，避免重复运行或覆盖结果。

CPU 分类头迁移检查：

```bash
python experiment/loveda_fewshot/check_mapped_head.py
```
