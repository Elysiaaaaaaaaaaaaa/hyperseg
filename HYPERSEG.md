# HyperSeg-UAV 代码讲解

本文档说明仓库中的两套 HyperSeg-UAV 实现：`hyperseg_uav/` 是保持原有训练、测试和推理接口的单图模型；新增的 `fewshot_hyperseg/` 是使用 support/query episode 和 HyperNetwork 的 few-shot 模型。

> 本文以代码现状为准。基础单图模型实现了 SegFormer-B3 多尺度特征、场景条件动态调制、动态融合和低秩适配器；few-shot 模型实现了 episode 内的掩码原型和 HyperNetwork，但两套模型都没有 prototype memory、在线记忆写入或形态学后处理。

## 1. 任务与类别

模型默认输出 9 个类别，标签定义如下：

| ID | 类别 |
|---:|---|
| 0 | Ignore |
| 1 | Background |
| 2 | Building |
| 3 | Road |
| 4 | Water |
| 5 | Barren |
| 6 | Vegetation |
| 7 | Agricultural |
| 8 | Vehicle |

类别 `0` 是忽略区域，不参与交叉熵、Dice、边界损失和 mIoU 计算。训练和评估脚本默认使用 `classes=9`、`ignore_index=0`。

## 2. 代码结构

```text
hyperseg_uav/
  data.py       # UAVDataset：读取、裁剪和增强图像/掩码
  model.py      # HyperSegUAV 及其编码器、动态模块和预测头
  losses.py     # hyperseg_loss 与 mean_iou
fewshot_hyperseg/
  fewshot_dataset.py  # episodic support/query 数据集
  model.py            # Few-shot + HyperNetwork 模型
  losses.py           # 分割、边界、原型和融合损失
tools/
  train_hyperseg.py  # 训练并保存最佳 checkpoint
  train_fewshot.py   # 训练 few-shot + HyperNetwork 模型
  test_hyperseg.py   # 在固定测试划分上计算 mIoU 和混淆矩阵
  infer_hyperseg.py  # 对任意 PNG 目录做滑窗推理并导出掩码
```

## 3. 数据读取：`UAVDataset`

`UAVDataset` 接收图像目录、可选掩码目录和图像 ID 列表。图像按文件名排序读取，仅处理 `.png` 文件；当提供 `ids` 时，既支持写入 stem（如 `test1_122`），也支持写入完整文件名。

### 训练模式

当 `training=True` 时，样本经过以下处理：

1. 从 `0.5、0.75、1.0、1.25、1.5` 中随机选择缩放比例，并保证缩放后的尺寸不小于 `size`。
2. 随机裁剪 `size x size` 区域，默认以 `scene_crop_prob=0.3` 的概率尝试包含至少 3 个非 Ignore 类别，或包含稀有类 `5、7、8` 的区域。
3. 以 0.5 概率水平翻转，以 0.5 概率垂直翻转。
4. 以 0.7 概率随机调整亮度和对比度，以 0.4 概率调整颜色。

图像使用双线性插值，掩码使用最近邻插值，以避免产生非法类别编号。图像转为 `[0, 1]` 的 `float32`，形状为 `C x H x W`；掩码转为 `int64`，形状为 `H x W`。

### 验证/测试模式

当 `training=False` 且 `size` 非空时，图像和掩码直接缩放到 `size x size`。因此 `tools/train_hyperseg.py` 和 `tools/test_hyperseg.py` 的验证图像默认是 `768 x 768`，而不是保留原始尺寸。

每个样本返回：

```python
{
    "image": Tensor[C, H, W],
    "name": "xxx.png",
    "original_size": (width, height),
    "mask": Tensor[H, W],  # 有 mask 时提供
}
```

## 4. 模型结构：`HyperSegUAV`

整体数据流如下：

```text
RGB image
   -> SegFormer-B3 encoder
   -> 4 级特征 [64, 128, 320, 512]
   -> SceneCondition 生成场景上下文 context[128]
   -> DynamicModulation 对各尺度做 FiLM 式调制
   -> DynamicFusion 按场景自适应融合多尺度特征
   -> DynamicLowRankAdapter 添加低秩残差
   -> 两层卷积 decoder
   -> segmentation head / boundary head
```

### 4.1 SegFormer-B3 编码器

`SegFormerB3Encoder` 使用 Hugging Face 的 `nvidia/mit-b3`，通过 `output_hidden_states=True` 取最后四级隐藏状态。某些 Transformers 版本返回 `B x HW x C` 的 token 格式，代码会根据平方根恢复空间尺寸并转换为 `B x C x H x W`。

构造模型时会优先读取项目内的 `models/nvidia--mit-b3/`，不存在时再使用 Hugging Face 模型名。`pretrained=True` 可从 Hugging Face 下载权重；推理脚本从 checkpoint 的配置重建模型后强制设为 `pretrained=False`，因此要求模型已经放入上述项目目录或存在于本机 Hugging Face 缓存中。训练时使用 `--no-pretrained` 可禁用在线下载，但仍需本地可加载模型结构。

### 4.2 场景条件与动态模块

`SceneCondition` 将每一级特征通过 `1x1` 卷积投影到 128 维，做全局平均池化后拼接，再经 MLP 得到场景上下文向量 `context`。

`DynamicModulation` 为每一级特征生成 `gamma` 和 `beta`，并执行：

```text
y = GroupNorm(x) * (1 + 0.15 * tanh(gamma)) + 0.15 * tanh(beta)
```

其线性层零初始化，因此模型初始时动态调制接近恒等行为，训练更稳定。

`DynamicFusion` 根据 `context` 生成 4 个 softmax 权重，将各尺度特征投影到 64 通道、上采样到最高分辨率后加权求和。`DynamicLowRankAdapter` 再由上下文生成低秩 down/up 矩阵，对融合特征增加一个受 `adapter_scale=0.1` 限制的残差。

### 4.3 输出

解码器由两组 `3x3 Conv + BatchNorm + GELU` 组成，随后有两个预测头：

- `head`：输出 `B x 9 x H x W` 的语义分割 logits；
- `boundary`：输出 `B x 1 x H x W` 的边界 logits，只有训练损失使用。

模型 `forward` 返回：

```python
{"logits": logits, "boundary": boundary, "context": context}
```

## 5. 损失与指标：`losses.py`

### 5.1 `hyperseg_loss`

损失只对非 Ignore 像素计算，默认由以下部分组成：

```text
total = 0.55 * focal_CE
      + 0.30 * mean_class_dice
      + 0.05 * 0.5 * rare_class_loss
      + 0.10 * boundary_BCE
```

- `focal_CE`：逐像素交叉熵乘以 `(1 - p_true)^1.5`，降低容易样本的影响；可传入 `class_weights`，训练脚本当前未传入，因此默认不使用类别权重。
- `mean_class_dice`：对实际出现在 batch 中的类别 `1..8` 计算 soft Dice 损失，再求平均；没有出现的类别不加入平均。
- `rare_class_loss`：对出现于目标中的 Barren（5）、Agricultural（7）、Vehicle（8）追加负对数概率惩罚。
- `boundary_BCE`：根据相邻像素标签是否不同构造边界目标，并与 `boundary` 头计算二元交叉熵。

最终通过 `torch.nan_to_num` 将 NaN 转为 0，避免异常数值中断训练。

### 5.2 `mean_iou`

指标先对 logits 做 `argmax`，再对类别 `1..8` 分别计算 IoU，忽略目标为 0 的像素。只要某类别在当前目标中有并集，就纳入平均；最终返回有效类别的平均 IoU，即 mIoU。

## 6. 训练流程：`train_hyperseg.py`

训练数据默认位于：

```text
dataset/train/train/images
dataset/train/train/masks
```

脚本从 `runs/splits/train.txt` 和 `runs/splits/val.txt` 读取样本 ID，构造训练集和验证集。默认超参数为：`60` 个 epoch、裁剪尺寸 `768`、batch size `2`、学习率 `1e-4`、随机种子 `3407`。

优化器是 AdamW，并对编码器使用 `lr * 0.1`，其余动态模块、解码器和预测头使用完整学习率。每个 batch 执行前向、损失计算、反向传播、梯度裁剪（范数上限 1.0）和参数更新；AMP 通过 `--amp` 开启，且只在 CUDA 上生效。

每轮结束后在验证集计算 mIoU。只有当当前 mIoU 高于历史最佳值时，才保存 checkpoint：

```python
{
    "model": model.state_dict(),
    "model_config": model.model_config,
    "epoch": epoch,
    "val_miou": score,
}
```

示例：

```bash
cd /home/apocalypse/code/Mondstadt
python tools/train_hyperseg.py \
  --data dataset \
  --split-dir runs/splits \
  --out runs/hyperseg_best.pt \
  --epochs 60 \
  --size 768 \
  --batch-size 2 \
  --amp
```

使用随机初始化时：

```bash
python tools/train_hyperseg.py --no-pretrained
```

## 7. 测试评估：`test_hyperseg.py`

测试脚本读取 `runs/splits/test.txt`，但仍使用带标签的训练目录：

```text
dataset/train/train/images
dataset/train/train/masks
```

它统计 `9 x 9` 混淆矩阵，忽略类别 0，输出总体 mIoU、每类 IoU 和混淆矩阵，并将同样内容写入 checkpoint 同目录下的 `<checkpoint_stem>_test_metrics.json`。

```bash
python tools/test_hyperseg.py \
  --checkpoint runs/hyperseg_best.pt \
  --split-dir runs/splits \
  --size 768
```

## 8. 推理导出：`infer_hyperseg.py`

推理脚本读取 checkpoint 中的 `model_config`，重建模型并加载 `model` 权重。它遍历 `--input` 目录下的所有 `.png`：

1. RGB 图像转为 `[0, 1]` 的张量。
2. 按 `size` 滑窗，步长为 `size * (1 - overlap)`。
3. 边缘不足一个窗口的区域先缩放到窗口大小，预测后再缩回原区域尺寸。
4. 重叠区域的 logits 取平均。
5. 对类别维做 `argmax`，保存为与输入同名的单通道 `uint8` PNG。

`--tta` 当前只做一次水平翻转：原图和水平翻转结果平均后再 argmax，并不是多模型集成。

```bash
python tools/infer_hyperseg.py \
  --checkpoint runs/hyperseg_best.pt \
  --input dataset/test_1/images \
  --output runs/predictions \
  --size 768 \
  --overlap 0.5 \
  --tta
```

输出目录会自动创建，输出文件名与输入文件保持一致。模型自动选择 CUDA；没有 CUDA 时使用 CPU。

## 9. 运行前检查

安装依赖：

```bash
python -m pip install -r requirements.txt
```

训练前确认以下文件存在：

```text
runs/splits/train.txt
runs/splits/val.txt
runs/splits/test.txt       # 运行 test_hyperseg.py 时需要
```

推理前必须已有训练生成的 checkpoint，并能在本地加载 `nvidia/mit-b3` 的 Transformers 模型文件。由于当前实现没有对输入目录为空、checkpoint 缺失或 PNG 尺寸/类别范围做额外提交校验，正式提交前应自行检查输出文件数量、尺寸、模式和像素值范围。

## 10. 一句话总结

HyperSeg-UAV 是“SegFormer-B3 多尺度编码器 + 场景上下文驱动的动态特征调制/融合 + 低秩适配器 + 语义与边界双头”的单模型分割方案；训练使用带稀有类别增强的监督损失，验证以忽略 Ignore 类别后的 mIoU 选择最佳 checkpoint，推理使用滑窗平均 logits 后导出类别掩码。

## 11. Few-shot + HyperNetwork 模型

`fewshot_hyperseg/` 是新增的 episodic 模型，与 `hyperseg_uav/` 的单图模型相互独立。每个 episode 包含 `N*K` 张带掩码的 support 图像和 `Q` 张 query 图像；模型在一次 SegFormer-B3 前向中编码两者，从 support 掩码提取每个类别的原型，再由 HyperNetwork 生成 query 的动态调制、融合和分支门控参数。

模型接口为：

```python
from fewshot_hyperseg import HyperSegUAV

output = model(
    query_images=query_images,       # [B, Q, 3, H, W]
    support_images=support_images,   # [B, N*K, 3, H, W]
    support_masks=support_masks,     # [B, N*K, H, W]
)
```

当 `Q > 1` 时，`output["logits"]` 的形状为 `[B, Q, 9, H, W]`；`Q == 1` 时会压缩为 `[B, 9, H, W]`。训练入口会自动把 support/query 图像调整为 `512 x 512`，把监督掩码调整为 `1024 x 1024`，并保持图像双线性、掩码最近邻插值。

训练数据仍使用：

```text
dataset/train/train/images
dataset/train/train/masks
```

示例命令：

```bash
python tools/train_fewshot.py \
  --dataset-root dataset \
  --output-dir runs/fewshot_hyperseg \
  --epochs 30 \
  --amp
```

该模型目前只有 episodic 训练入口；现有 `tools/infer_hyperseg.py` 仍对应单图 `hyperseg_uav` 模型，不能直接加载 few-shot 模型 checkpoint，因为后者推理时必须提供 support 图像和 support 掩码。
