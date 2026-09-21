# HyperSeg-UAV 项目协作说明

## 项目概览

这是一个基于 PyTorch 的无人机航拍语义分割项目。模型输入 RGB 图像，输出 9 类像素级预测；类别 `0` 为 Ignore，不参与主要损失和 mIoU 计算。

本地没有充足算力，项目需要部署在服务器上运行
时间较长，训练轮数多的训练任务在服务器上跑需要使用后台命令启动，日志打印到log文件里。训练结束后将log文件下载到对应实验文件夹下。

核心目录：

- `hyperseg_uav/`：数据集、模型、损失函数和指标。
- `tools/`：训练、测试、滑窗推理、可视化和提交检查脚本。
- `HYPERSEG.md`：当前实现的中文说明，以代码现状为准。
- `experiment/`：实验相关脚本、日志、说明文档。每个实验独立有一个文件夹


## 开发约定

- 默认使用 Python、PyTorch 和 Transformers；模型骨干为 Hugging Face `nvidia/mit-b3`。
- 修改代码后优先做轻量级语法检查或针对性测试；完整训练需要本地数据集、权重缓存和较高显存，不应在没有这些资源时直接启动。
- 不要提交数据集、模型权重、预测结果或缓存目录。训练输出默认放在 `runs/`，数据默认放在 `dataset/`。
- 保持标签约定：类别数默认为 `9`，`ignore_index=0`，有效类别为 `1..8`。
- 图像和掩码处理必须保持空间对齐；图像缩放使用双线性插值，掩码缩放使用最近邻插值。
- 不要为了修复无关问题大范围重写现有实现；先阅读 `HYPERSEG.md` 和相关脚本，再进行最小必要修改。
- 进行`experiment/`中的实验后，要分析新的日志并将分析加入该实验文件夹下的总研究日志`experiment.md`中

### 数据集约定（数据集在服务器上路径与称呼对照表）

当前主数据集是 `low_altitude_2026`。本地代码目录为
`/mnt/d/myproject/hyperseg`，server2 代码目录为
`/root/autodl-tmp/hyperseg`；server2 上的数据与代码分开存放。

| 称呼 | 本地路径 | server2 路径 | 内容与用途 |
| --- | --- | --- | --- |
| 有标注训练图像 | `dataset/low_altitude_2026/train/images` | `/root/autodl-tmp/data/low_altitude_2026/train/images` | 5598 张训练图片；与同名 `masks` 配对 |
| 有标注训练掩码 | `dataset/low_altitude_2026/train/masks` | `/root/autodl-tmp/data/low_altitude_2026/train/masks` | 单通道标签，像素值 `0..8`；0 为 Ignore |
| 固定训练划分 | `runs/splits/train.txt` | `/root/autodl-tmp/hyperseg/runs/splits/train.txt` | 5598 个样本 ID/stem，只用于训练 |
| 固定验证划分 | `runs/splits/val.txt` | `/root/autodl-tmp/hyperseg/runs/splits/val.txt` | 699 个样本 ID/stem，用于选最佳 checkpoint |
| 固定有标注测试划分 | `runs/splits/test.txt` | `/root/autodl-tmp/hyperseg/runs/splits/test.txt` | 699 个有标注样本；只用于最终离线 mIoU，不是比赛无标注测试集 |
| 无标注比赛测试图像 | `dataset/low_altitude_2026/images` | `/root/autodl-tmp/data/low_altitude_2026/images` | 500 张图片；没有掩码，只用于导出提交预测 |

数据集的两个“test”概念必须区分：`runs/splits/test.txt` 是从有标注训练数据中固定划出的离线评估集；`low_altitude_2026/images` 是独立的无标注比赛测试集，不能用它计算 mIoU。提交预测必须与无标注输入同名，并通过 `tools/check_submission.py` 的单通道、1024×1024 和 `0..8` 检查。

兼容旧脚本时，带标签目录解析器还可能接受以下布局：

```text
DATA_ROOT/train/images
DATA_ROOT/train/masks
DATA_ROOT/train/train/images
DATA_ROOT/train/train/masks
DATA_ROOT/low_altitude_2026/train/images
DATA_ROOT/low_altitude_2026/train/masks
```

在当前实际数据中，优先使用上表的 `low_altitude_2026/train/{images,masks}`。不要把 `images`（无标签比赛图像）当作训练图像，也不要把 `test.txt` 的有标注测试划分改写成无标签测试目录。划分文件每行使用不带 `.png` 的 stem，例如 `test1_123`；图像和掩码文件名必须完全对应。

## 常用命令

从项目根目录运行：

```powershell
python tools/train_hyperseg.py --data dataset --split-dir runs/splits --out runs/hyperseg_best.pt
python tools/test_hyperseg.py --checkpoint runs/hyperseg_best.pt --split-dir runs/splits
python tools/infer_hyperseg.py --checkpoint runs/hyperseg_best.pt --input dataset/test_1/images --output runs/predictions --tta
python tools/visualize_predictions.py --predictions runs/predictions --images dataset/test_1/images --output runs/visualized
python tools/check_submission.py runs/predictions dataset/test_1/images
```

运行前确认数据目录和 `runs/splits/{train,val,test}.txt` 存在，并确保本地可以加载 `nvidia/mit-b3` 的模型配置和权重。

## 验证重点

- 检查导出的预测文件名是否与输入图像完全一致。
- 提交检查要求预测为单通道 `L` 模式、`1024x1024`，像素值在 `0..8`。
- 评估时关注 Ignore 区域处理、类别编号、掩码尺寸以及 checkpoint 中的 `model_config`。
- 当前代码没有实现 prototype memory、在线记忆写入和三视图 TTA；不要将说明文档中的计划功能当作已有功能。

## 当前仓库状态

当前工作目录未检测到 Git 仓库；如需版本控制或分支操作，先确认用户希望如何初始化和管理 Git。
