# MathSeg-UAV：M0～M4 受控实验

本目录实现《MathSeg_UAV_数学解析无人机语义分割方案.md》的第一阶段实验。
目标是验证数学空间先验对融合的贡献。尚未实现域对齐、稀疏动态计算或轻量骨干实验，
也不预设 mIoU 或速度提升。所有新代码独立于现有 HyperSeg、Mask2Former 实验。

## 模型与消融

| 实验 | 融合方式 | 数学先验 | 损失 |
| --- | --- | --- | --- |
| M0 | 四级特征固定平均 | 不计算 | CE + Dice + 0.1 × Boundary BCE |
| M1 | 固定平均后残差调制 | 使用 | 同 M0 |
| M2 | 语义特征预测逐位置尺度权重 | 计算后置零 | 同 M0 |
| M3 | 语义特征和先验联合预测尺度权重 | 使用 | 同 M0 |
| M4 | 同 M3 | 使用 | CE 加入渐进类别权重，其他项不变 |

- 骨干：默认 Hugging Face `nvidia/mit-b3`，ImageNet 均值/标准差归一化。
- 四级特征分别投影到 64 通道，对齐到第一层特征的空间大小，通常为输入的 1/4。
- 固定算子作用于**完成数据增强、仍处于 [0,1] 的 RGB**，转换灰度后计算
  Sobel 幅值、3×3 和 7×7 局部方差、Laplacian 绝对值，共 4 个通道。
- 算子使用 FP32 和 replicate padding。各先验按每图每通道的
  `x / (mean(x) + x + 1e-6)` 归一化，然后双线性对齐特征；归一化前将不超过
  `1e-7` 的响应归零，避免把常色区域的 FP32 舍入误差放大。
- M1 使用 `F * (1 + 0.1 * tanh(Conv(M)))`，卷积零初始化。
- M2/M3/M4 使用同形状的 `1×1 Conv → GELU → 1×1 Conv → 尺度维 softmax`。
  四个位置权重之和为 1；最后一层零初始化，初始为平均融合。
- M2 与 M3 的可学习参数数量完全一致，同一种子下初始权重一致；M2 的先验输入置零，
  所以对应连接不会获得训练梯度。保留先验计算，使 M2/M3 算子开销相近。
- 分割头输出 9 类，另有训练用边界头；统一使用 GroupNorm，避免 batch 大小影响 BN 统计。
- 第一轮不引入原 HyperSeg 的 Focal 或稀有类附加惩罚。这里的 M0 是新实验自己的受控基线，
  **不是原 HyperSeg，也不是标准 SegFormerHead**。

最重要的对比是 M3/M2，其次是 M1/M0。M4/M3 检查类别平衡。
这是稠密融合模型，新增模块不会自动减少骨干计算量。

## 模型结构详解：与原 HyperSeg 的关系

这里的“原 HyperSeg”特指本仓库 [hyperseg_uav/model.py](../../hyperseg_uav/model.py)
中的单图 `HyperSegUAV`，不指 `fewshot_hyperseg` 或 Swin-L 版本。
MathSeg 实现位于 [model.py](model.py)，顶层类为 `MathSegUAV`。

当前 M0～M4 是一套独立构建的受控模型：沿用原 HyperSeg 的 **MiT-B3 四级特征、
通道投影、融合后卷积解码、语义/边界双头**这一基本设计，重新实现融合与先验模块。
它们没有实例化原 `HyperSegUAV`，也没有直接调用原模型的动态模块。
默认加载的是 MiT-B3 的骨干预训练权重，不是已经训练好的 HyperSeg checkpoint；
新实验的投影层、解码器、预测头和 gate 都在本次实验中训练。

### 1. 原 HyperSeg 的完整路径

```text
RGB [B,3,H,W]
  → SegFormerB3Encoder：提取 F1、F2、F3、F4
  → SceneCondition：各层投影、全局平均池化、拼接、MLP
      得到整图场景向量 context [B,128]
  → DynamicModulation：由 context 生成各层通道的 gamma/beta
      对四级特征分别进行 GroupNorm + FiLM 式调制
  → DynamicFusion：由 context 生成整图共用的四个尺度权重 [B,4]
      各层 1×1 投影、双线性对齐，再加权求和
  → DynamicLowRankAdapter：由 context 生成低秩矩阵，添加融合特征残差
  → 两层 3×3 Conv + BatchNorm + GELU
  → 1×1 语义头 / 1×1 边界头
  → 双线性上采样到输入尺寸
```

`SceneCondition` 的 context 同时供调制、融合和低秩适配器使用，并不是仅用于融合。
原 `DynamicFusion` 的权重可以随图片变化，但一张图内所有空间位置共用同一组权重。

### 2. 哪些沿用，哪些改写，哪些是新增模块

| 部分 | 原 HyperSeg | MathSeg M0～M4 | 关系与代码位置 |
| --- | --- | --- | --- |
| 骨干网络 | `SegFormerB3Encoder` 包装 HF `SegformerModel` | `MathSegUAV.encoder` 直接构建 HF `SegformerModel` | 沿用同类 MiT-B3 骨干，未复用原包装类；保存完整 encoder config |
| 输入归一化 | 原单图路径将数据集的 `[0,1]` 图像直接送入编码器 | 模型内显式执行 ImageNet mean/std 归一化 | 新实验明确了预处理；两套历史成绩不能只按骨干名称直接比较 |
| 四级通道投影 | `DynamicFusion.projections` 的 1×1 卷积 | `FusionHead.projections` 的 1×1 卷积 | 沿用设计，独立参数；默认都投影到 64 通道 |
| 尺寸对齐 | 双线性对齐到 F1 大小 | 同样对齐到 F1 大小 | 沿用操作方式，`align_corners=False` |
| 场景向量 | `SceneCondition`，整图 context | 没有该模块 | M0～M4 均未保留 |
| 四级特征动态调制 | `DynamicModulation`，由 context 生成通道 gamma/beta | 没有该模块；M1 有独立的先验调制 | M1 的条件、作用位置和公式均与原模块不同 |
| 融合权重 | `DynamicFusion.weights`，输出 `[B,4]` | M0/M1 固定平均；M2/M3/M4 的 `FusionHead.gate` 输出空间权重 | MathSeg 新实现；权重形状为 `[B,4,H/4,W/4]` |
| 动态低秩适配器 | `DynamicLowRankAdapter` | 没有该模块 | M0～M4 均未保留 |
| 融合后解码器 | 两层 `3×3 Conv + BatchNorm + GELU` | 两层 `3×3 Conv + GroupNorm(1,64) + GELU` | 沿用两层卷积设计，但替换归一化；代码和参数独立 |
| 语义/边界双头 | 64→9 和 64→1 的 1×1 卷积 | `FusionHead.segmentation` 和 `.boundary` | 沿用双头设计，参数独立训练 |
| 数学空间先验 | 没有 | `SpatialPrior` | MathSeg 新增的固定算子分支 |
| 类别渐进权重 | 原损失有 Focal 和稀有类附加项 | M4 的 `class_weights()` | 新实验的训练策略，不是网络层 |
| 数据增强 | `UAVDataset` | `CheckedDataset(UAVDataset)` | 这里确实直接继承了原数据集代码，增加标签检查及可恢复采样 |

因此，“沿用设计”不等于“加载了原 HyperSeg 的对应层或权重”。模型代码没有导入
`hyperseg_uav.model`；直接复用发生在数据集层。损失也由本目录
[protocol.py](protocol.py) 独立实现，没有直接调用原 `hyperseg_loss`。

### 3. 五个实验共用的骨干、投影和解码部分

以下使用默认 `width=64`、四个先验通道。`B` 表示 batch size；表中尺寸对应
512×512 输入，1024×1024 输入时各层空间边长翻倍。一般尺寸以骨干实际输出为准。

| 张量/层 | 默认形状 | 含义 |
| --- | --- | --- |
| 输入 RGB | `[B,3,512,512]` | 增强后的图像，范围 `[0,1]` |
| F1 | `[B,64,128,128]` | MiT-B3 第一阶段，约 1/4 分辨率 |
| F2 | `[B,128,64,64]` | 第二阶段，约 1/8 分辨率 |
| F3 | `[B,320,32,32]` | 第三阶段，约 1/16 分辨率 |
| F4 | `[B,512,16,16]` | 第四阶段，约 1/32 分辨率 |
| P1～P4 | 各 `[B,64,128,128]` | 四级特征经独立 1×1 投影并上采样对齐 |
| 融合特征 Z | `[B,64,128,128]` | 固定平均或逐位置加权求和 |
| 解码特征 D | `[B,64,128,128]` | Z 经两层卷积解码器 |
| 低分辨率语义/边界 logits | `[B,9,128,128]` / `[B,1,128,128]` | 两个独立的 1×1 预测头 |
| 最终语义/边界 logits | `[B,9,512,512]` / `[B,1,512,512]` | 双线性恢复输入空间尺寸 |

所有实验的 `P1～P4` 都来自未经原 HyperSeg 场景调制的骨干特征。
解码器的两层卷积均为 64→64、kernel 3×3、padding 1、不带 bias；
`GroupNorm(1,64)` 表示一个归一化组，并不是每通道各一个组。
语义输出取 `argmax` 后得到类别图，边界输出只提供辅助监督，不参与语义 logits 的融合。
当前 forward 在推理时仍计算边界头，因此测速也包括这部分开销。

### 4. MathSeg 新增分支：SpatialPrior

数学先验分支与骨干并行读取同一张增强后的图像，先验本身不作为额外通道输入 MiT-B3：

```text
增强后的 RGB [0,1]
  ├→ ImageNet 归一化 → MiT-B3 → P1、P2、P3、P4
  └→ 灰度图 → 固定数学算子 → 数值归一化 → 对齐到 F1 → M
```

灰度图为 `G = 0.299R + 0.587G_rgb + 0.114B_rgb`。默认先验顺序如下：

| M 的通道 | 计算方式 | 要描述的图像属性 |
| --- | --- | --- |
| 0：gradient | `sqrt(Sobel_x(G)^2 + Sobel_y(G)^2)` | 一阶边缘变化强度 |
| 1：variance3 | `AvgPool3(G²) - AvgPool3(G)²`，负数截为 0 | 较小窗口的局部纹理变化 |
| 2：variance7 | `AvgPool7(G²) - AvgPool7(G)²`，负数截为 0 | 较大窗口的局部纹理变化 |
| 3：laplacian | 四邻域 Laplacian 响应的绝对值 | 二阶空间变化强度 |

这些算子在输入分辨率计算，完成前文所述数值处理后再对齐到 F1。
默认 `M` 的形状为 `[B,4,128,128]`，**四个先验通道不是四个尺度的融合权重**；
从先验到尺度权重的映射由后续 gate 学习。
固定卷积核通过 buffer 保存，没有可训练参数；训练不需要为先验单独提供标签。
梯度或纹理强并不必然代表小目标，因此 M3 将 M 与语义特征一起使用，不把梯度直接当作目标尺寸。

### 5. M0：固定平均融合基线

```text
MiT-B3 → P1、P2、P3、P4 → 固定平均 Z → 公共解码器 → 双头
```

融合公式为 `Z = (P1 + P2 + P3 + P4) / 4`，权重恒为 0.25。
M0 不执行 Sobel、方差或 Laplacian 运算；为保持返回接口一致，只生成零值 prior 张量。
它没有场景 context、FiLM、低秩适配器，也没有学习尺度权重的 gate。

M0 用来确定这套公共骨干和解码器的基础表现。它与原 HyperSeg 存在上述多处结构及训练差异，
不能将“原 HyperSeg 的旧分数”直接填进 M0 一行。

### 6. M1：数学先验调制融合特征

```text
MiT-B3 → P1、P2、P3、P4 → 固定平均 Z ──────────┐
RGB → SpatialPrior → M → 1×1 Conv(4→64) → tanh ─┤
                                              ↓
                          Z' = Z × (1 + 0.1 × tanh(Conv(M)))
                                              ↓
                                      公共解码器 → 双头
```

这里的乘法按通道和位置逐元素执行。M1 先以相同比例融合四个尺度，再用 M 调整融合特征。
因调制范围受 `0.1 × tanh` 限制，特征乘数在约 0.9～1.1 之间；
调制卷积权重和偏置均零初始化，初始行为与固定平均融合一致。

它与原 `DynamicModulation` 的差别是：原模块使用整图 context，在融合前调制每一级特征，
同时包含乘性 gamma 和加性 beta；M1 使用局部数学先验，在融合后只进行乘性残差调制。
M1/M0 验证“先验用于调整特征是否有帮助”，不验证自适应尺度选择。

### 7. M2：仅由语义特征决定逐位置尺度权重

```text
MiT-B3 → P1、P2、P3、P4 ──────────────────────────┐
RGB → SpatialPrior → M → 置零后的 4 通道 ─────────┤
                                                ↓
                       拼接 U [B,260,128,128]
                         → 1×1 Conv(260→64) → GELU
                         → 1×1 Conv(64→4) → 尺度维 softmax
                                                ↓
                             A [B,4,128,128]
                                                ↓
                        Z = Σ A_k × P_k → 公共解码器 → 双头
```

260 个输入通道来自 `4×64` 个投影特征通道和 4 个零值先验占位通道。
每个位置都得到一组四尺度权重，且 `Σ_k A_k(x,y)=1`；同一个尺度的权重在该位置的
64 个特征通道之间共享。gate 是共享参数的 1×1 卷积网络，不是为每个像素生成一套卷积参数。

M2 不使用有效的数学先验信息。先计算 M 再置零，是为了与 M3 保持相同 gate 输入形状和相近
先验算子开销。其作用是排除“只要把整图/固定权重改为空间权重就能提升”的解释。

### 8. M3：数学先验引导的逐位置尺度融合

M3 与 M2 使用完全相同的 gate，将拼接输入中的零通道换成真实 M：

```text
U = Concat(P1, P2, P3, P4, M)       # [B,260,128,128]
A = Softmax_scale(Conv2(GELU(Conv1(U))))
Z = A1×P1 + A2×P2 + A3×P3 + A4×P4
Z → 公共解码器 → 双头
```

这是当前实现中对应方案文档“数学尺度场模块 MSF”的部分，代码名称是
`SpatialPrior + FusionHead.gate`，没有另外名为 `MSF` 的类。
模型学习的是“给定语义特征和局部数学变化，各尺度应该占多大比例”，
并没有显式预测一个带真实尺度单位的标量场 `s(x,y)`。

M3 **没有叠加 M1 的调制卷积**。M1 是并行的先验使用方式对照，M3 则直接把先验用于尺度选择。
因此不能把 M0→M1→M2→M3 解读为逐个叠加全部前序模块；关键实验关系是
M1 对 M0、M3 对 M2。M3 与 M2 末层 gate 均零初始化，起点都是均匀尺度融合。

### 9. M4：M3 的模型结构，加上渐进类别重加权

M4 的 encoder、SpatialPrior、投影、gate、decoder 和双头与 M3 一致。
区别只在训练入口为 CE 提供由训练集频率和训练进度计算的类别权重：

```text
训练集掩码 → 统计各类频率 ─┐
当前 update / 总 update ──┴→ class_weights → 加权 CE
RGB → 与 M3 相同的网络 → 语义 logits / 边界 logits
                             ↓
                     加权 CE + Dice + 0.1×Boundary BCE
```

类别权重不接入 forward，不增加推理层数或模型参数，也不在推理时读取标签或类别频率。
训练后 M3 与 M4 的参数数值通常不同，但结构相同；M4/M3 衡量的是重加权训练策略的效果。

### 10. 默认非骨干参数量与实验结论边界

以下仅统计 `FusionHead` 可训练参数，按默认 `width=64`、4 个先验通道计算，
不包含 MiT-B3。SpatialPrior 的固定核、RGB 均值和标准差均为 buffer，不计入参数量。

| 实验 | 公共投影、解码器和双头 | 额外可训练模块 | 非骨干参数合计 |
| --- | ---: | --- | ---: |
| M0 | 140,426 | 无 | 140,426 |
| M1 | 140,426 | 先验调制 1×1 卷积：320 | 140,746 |
| M2 | 140,426 | 尺度 gate：16,964 | 157,390 |
| M3 | 140,426 | 尺度 gate：16,964 | 157,390 |
| M4 | 140,426 | 尺度 gate：16,964 | 157,390 |

参数计数可以在 `FusionHead` 或训练输出的配置中核对；训练配置中的 `parameters` 是
包含骨干的总数。改变 width 或先验通道数后，上表不再适用。

M0～M4 内部共用新的归一化、GroupNorm 解码器和基础损失，适合进行上述受控比较。
若要声称优于原完整 HyperSeg，需要额外按匹配的数据划分、训练预算和原图评估协议运行原模型，
并明确两者的预处理和损失差异。当前脚本没有把“完整 HyperSeg + MathSeg 模块”作为一个额外变体。

## 数据和指标

支持以下数据布局，也可以显式传入 `--image-dir` 和 `--mask-dir`：

```text
DATA_ROOT/train/{images,masks}
DATA_ROOT/train/train/{images,masks}
DATA_ROOT/low_altitude_2026/train/{images,masks}
```

使用 `runs/splits/{train,val,test}.txt`。当前固定划分为 5598/699/699，
脚本不硬编码样本数。条目支持 stem 或 `.png` 文件名；规范化后检查重复、交集、缺失文件。
训练运行前建议扫描全部标签和图像/掩码尺寸：

```bash
python experiment/mathseg_uav/train.py --check-only \
  --data-root /root/autodl-tmp/data/low_altitude_2026 \
  --split-dir runs/splits
```

标签必须为单通道整数 `0..8`，0 是 Ignore。几何增强复用 `UAVDataset`：图像双线性、
标签最近邻。每个样本首次在进程内读取时检查原始尺寸、标签范围。
如果调色板掩码转 L 会改变类别编号，脚本会明确拒绝读取，需先将原始调色板索引
保存为 L 标签图；不能直接对彩色调色板调用 `convert('L')`。
检查 ID 不能识别来自同一场景的相邻影像，场景泄漏仍需数据来源信息审计。

正式验证/测试固定为原始分辨率、batch 1、整图 FP32、无 TTA：

- 全数据集累计 9×9 混淆矩阵；只对 GT 非零像素计数。
- mIoU 为类 1..8 中有并集类别的均值。无并集类别记为 JSON `null`。
- 有效像素被预测成 0 仍计为该真实类别的错误；不会把该像素丢弃。
- 附加 `boundary_f1_exact`：原始像素坐标下的双侧语义边界 micro F1，无容差匹配。
  不能直接与使用不同边界容差的论文分数比较。
- 边界来自相邻标签是否不同，不对类别编号求数值梯度；Ignore 的 3×3 邻域不参与边界损失/指标。
- 所有 Ignore batch 的损失为可反向传播的 0；非有限损失/梯度立即报错，不用 `nan_to_num` 隐藏异常。

M4 只扫描 **train** 掩码统计频率。`gamma = gamma_max * update / max_updates`，默认上限 0.5。
有效已出现类的逆频率权重先均值归一化，再限幅到 `[0.25,4]`，Ignore 权重为 0；
加权 CE 按有效像素权重和归一化。不使用验证集 IoU、测试集频率或伪标签。

训练脚本支持 `--val-size N` 仅用于服务器短程 smoke test，把验证图像缩放到 `N×N`；
默认值 0 保持原始分辨率。正式 M0/M1 训练不要传这个参数，或显式使用 `--val-size 0`。

## 环境

使用已有服务器的 PyTorch/Transformers 环境。训练入口使用 `torch.amp.GradScaler`，
需要 PyTorch 2.3 或更新版本；Transformers 使用项目的 4.x 系列。
本地 CPU 回归测试环境为 Python 3.8、PyTorch 2.4.1+cpu、Transformers 4.46.3、
NumPy 1.24.4、Pillow 10.4.0。未在本次实现中验证服务器 CUDA 运行。

默认只读本地权重/缓存。可以将 Hugging Face 模型目录放在
`models/nvidia--mit-b3/`，或通过 `--model-name /path/to/hf_mit_b3` 指定。
确需联网加载时传 `--allow-download`。`--no-pretrained` 是真正的随机初始化，
但仍需要所选骨干的 config；不要与预训练实验混为一组。

Checkpoint 包含完整 encoder config，恢复、评估和导出不需要再次下载骨干。
只加载自己信任的 checkpoint，因为恢复优化器和 RNG 状态使用 pickle。

## 先检查，再在服务器训练

下面的 `plan` 只打印命令，不需要导入 PyTorch，也不启动训练：

```bash
python experiment/mathseg_uav/run.py plan \
  --variants M0 M1 M2 M3 --seeds 3407 \
  --output-root runs/mathseg_uav/primary -- \
  --data-root /root/autodl-tmp/data/low_altitude_2026 \
  --model-name /path/to/hf_mit_b3
```

队列参数放在 `--` 前，训练参数放在 `--` 后。
默认 512 裁剪、batch 2、累积 1、160000 次参数更新、FP32、AdamW，学习率 `1e-4`，
骨干倍率 0.1、weight decay 0.01、PolyLR power 0.9、梯度裁剪 1.0。
每 2800 次更新验证并存盘；`--save-interval` 可缩短最近断点的保存间隔。
有效 batch 为 `batch-size × accumulation`，更改累积数不会更改 `max-updates` 的含义。
所有组使用相同参数、精度和预算。该初始协议不等同于已有 H1/H3 的全部训练设置。

先在服务器做 20 次更新检查（输出独立于正式训练）：

```bash
python experiment/mathseg_uav/train.py \
  --variant M3 --data-root /root/autodl-tmp/data/low_altitude_2026 \
  --model-name /path/to/hf_mit_b3 \
  --work-dir runs/mathseg_uav/smoke/M3_seed3407 \
  --max-updates 20 --val-interval 20 --save-interval 10 --log-interval 1
```

正式长训练使用后台队列，依次运行避免抢占同一张 GPU：

```bash
PYTHON_BIN=/path/to/server/env/bin/python \
bash experiment/mathseg_uav/launch.sh runs/mathseg_uav/primary \
  --variants M0 M1 M2 M3 --seeds 3407 -- \
  --data-root /root/autodl-tmp/data/low_altitude_2026 \
  --model-name /path/to/hf_mit_b3
```

`queue.log` 记录队列进度，`queue.pid` 是本地服务器上的后台 PID。
每组自己的 `train.log` 记录训练输出，`exit_code` 记录退出状态；任一组失败会停止队列。
`queue.lock` 防止在同一输出根目录并发启动队列。若机器异常退出留下锁，先确认对应 PID
不再运行，再手动删除该**单个锁文件**后恢复。直接调用 train.py 时不要把多个进程写入同一目录。

需要 M4 时将 `--variants` 设为 `M0 M1 M2 M3 M4`；已有 M0～M3 可以加 `--resume`
跳过完成项并继续剩余组。队列的 `--resume` 假定沿用原参数，改变实验配置应另开输出根目录。
正式复核可用 `--seeds 3407 2027 42`；默认不自动启动三个种子。

### server2 的实际命令

server2 的代码目录是 `/root/autodl-tmp/hyperseg`，有标注数据是
`/root/autodl-tmp/data/low_altitude_2026/train/{images,masks}`，固定划分位于代码目录的
`runs/splits`，MiT-B3 本地目录为 `/root/autodl-tmp/hyperseg/models/nvidia--mit-b3`。
进入服务器后先执行只读数据和标签检查：

```bash
cd /root/autodl-tmp/hyperseg
.venv-mask2former/bin/python -u experiment/mathseg_uav/train.py \
  --check-only \
  --data-root /root/autodl-tmp/data/low_altitude_2026 \
  --split-dir /root/autodl-tmp/hyperseg/runs/splits
```

该检查会读取三个划分的所有图像和掩码，确认 stem、尺寸、L 模式和标签 `0..8`；
首次运行可能需要几分钟。若只做代码/权重导入检查，可执行：

```bash
.venv-mask2former/bin/python - <<'PY'
from experiment.mathseg_uav.model import MathSegUAV
for variant in ("M0", "M1"):
    model = MathSegUAV(
        variant=variant,
        model_name="/root/autodl-tmp/hyperseg/models/nvidia--mit-b3",
        pretrained=True, local_files_only=True,
    )
    print(variant, sum(p.numel() for p in model.parameters()))
PY
```

分配到 GPU 后，先用同一环境做 20 次更新的短程检查；`--val-size 256` 只为缩短验证，
不能作为正式成绩：

```bash
for variant in M0 M1; do
  mkdir -p /root/autodl-tmp/work_dirs/mathseg_uav/smoke/${variant}_seed3407
  .venv-mask2former/bin/python -u experiment/mathseg_uav/train.py \
    --variant "$variant" \
    --data-root /root/autodl-tmp/data/low_altitude_2026 \
    --split-dir /root/autodl-tmp/hyperseg/runs/splits \
    --model-name /root/autodl-tmp/hyperseg/models/nvidia--mit-b3 \
    --work-dir /root/autodl-tmp/work_dirs/mathseg_uav/smoke/${variant}_seed3407 \
    --max-updates 20 --val-interval 20 --save-interval 10 --log-interval 1 \
    --val-size 256 --num-workers 4 \
    > /root/autodl-tmp/work_dirs/mathseg_uav/smoke/${variant}_seed3407/train.log 2>&1
done
```

正式 M0/M1 队列使用原始分辨率验证。下面的 `launch.sh` 会让 M0 完成后再开始 M1，
并把每组日志写到各自目录；请先确认当前作业分配了 CUDA GPU，再执行：

```bash
cd /root/autodl-tmp/hyperseg
PYTHON_BIN=/root/autodl-tmp/hyperseg/.venv-mask2former/bin/python \
bash experiment/mathseg_uav/launch.sh \
  /root/autodl-tmp/work_dirs/mathseg_uav/m0_m1_seed3407 \
  --variants M0 M1 --seeds 3407 -- \
  --data-root /root/autodl-tmp/data/low_altitude_2026 \
  --split-dir /root/autodl-tmp/hyperseg/runs/splits \
  --model-name /root/autodl-tmp/hyperseg/models/nvidia--mit-b3 \
  --size 512 --batch-size 2 --accumulation 1 \
  --max-updates 160000 --val-interval 2800 --save-interval 2800 \
  --num-workers 4 --device cuda
```

检查队列和 GPU：

```bash
cat /root/autodl-tmp/work_dirs/mathseg_uav/m0_m1_seed3407/queue.pid
tail -f /root/autodl-tmp/work_dirs/mathseg_uav/m0_m1_seed3407/queue.log
nvidia-smi
```

当前通过 SSH 检查到的 server2 会话暂时显示 `nvidia-smi: No devices found`，因此本次只完成了
脚本同步、数据路径确认、模型导入和 CPU 前向检查，没有启动训练。提交长任务前必须确认 GPU
已经分配；没有 GPU 时脚本会在 `--device cuda` 处直接退出，不会误用 CPU 跑长实验。

## 断点恢复

每组保存 `best.pt`（验证选择）和 `last.pt`（最近训练断点），包含模型、优化器、AMP scaler、
学习率总预算、随机状态、训练位置、模型配置、划分哈希、M4 类别统计。
训练按 epoch 和样本索引固定增强随机种子，从下一 microbatch 恢复，不受预取顺序影响。
CPU 回归测试检查连续训练与中断恢复的最终模型逐位一致；GPU 不承诺跨环境逐位一致。

恢复队列时，使用原命令并在 `--` 前增加 `--resume`。直接恢复示例：

```bash
python experiment/mathseg_uav/train.py \
  --resume runs/mathseg_uav/primary/M3_seed3407/last.pt \
  --data-root /root/autodl-tmp/data/low_altitude_2026
```

上例仅适用于原训练使用默认超参数。非默认的 crop、batch、累积、学习率、seed、先验设置
等需一并重复传入；脚本会拒绝与 checkpoint 不一致的训练协议。恢复时自动读取 variant。
允许更改数据路径和 worker 数，但 split 哈希必须一致；哈希仅验证 ID 划分，不验证图片文件内容。

`--stop-after 20 --max-updates 160000` 可在保持正式学习率总预算的前提下暂停并保存，
再按相同 `max-updates` 恢复。`--max-updates 20` 则是独立的短训练预算，两者不能混用。

## 单项先验与空间打乱对照

```bash
python experiment/mathseg_uav/train.py --variant M3 \
  --prior-channels gradient \
  --work-dir runs/mathseg_uav/prior_gradient/M3_seed3407

python experiment/mathseg_uav/train.py --variant M3 \
  --prior-mode shuffle \
  --work-dir runs/mathseg_uav/prior_shuffle/M3_seed3407
```

这些例子使用默认本地数据路径，服务器请补充相同数据/权重参数。
可选通道为 `gradient variance3 variance7 laplacian`。
删通道会改变 gate 参数量；严格的等参数对照用 `--prior-mode zero` 或 `shuffle` 保留通道数。
shuffle 在特征网格上使用固定种子的像素置换，各先验通道共用置换，不引入验证随机性。

## 测试、导出、测速与汇总

固定有标签 test 集只在选定方法后评估，不用它调整超参数：

```bash
python experiment/mathseg_uav/train.py \
  --eval-checkpoint runs/mathseg_uav/primary/M3_seed3407/best.pt \
  --eval-split test --data-root /root/autodl-tmp/data/low_altitude_2026
```

快速扫描 M1 的数学先验调制强度（不训练、不修改 checkpoint）：

```bash
python experiment/mathseg_uav/diagnose_modulation.py \
  --checkpoint runs/mathseg_uav/primary/M1_seed3407/best.pt \
  --data-root /root/autodl-tmp/data/low_altitude_2026 \
  --split-dir /root/autodl-tmp/hyperseg/runs/splits \
  --split val \
  --factors 0 0.1 0.3 0.5 1.0 1.5 \
  --output runs/mathseg_uav/primary/M1_seed3407/modulation_diagnostic.json
```

脚本以 `0.1` 作为参考，输出每个系数的 mIoU、类别 IoU、边界 F1、预测变化比例和实际调制幅度。

无标签比赛图像导出，默认要求 1024×1024，输出目录必须为空：

```bash
python experiment/mathseg_uav/inference.py export \
  --checkpoint runs/mathseg_uav/primary/M3_seed3407/best.pt \
  --input /root/autodl-tmp/data/low_altitude_2026/images \
  --output runs/mathseg_uav/primary/M3_seed3407/predictions
python tools/check_submission.py \
  runs/mathseg_uav/primary/M3_seed3407/predictions \
  /root/autodl-tmp/data/low_altitude_2026/images
```

导出保持原文件名、单通道 L 模式、原分辨率及 `0..8` 标签值。无 TTA。
模型配置从 checkpoint 恢复，不能用旧的 `infer_hyperseg.py` 加载这些权重。

在空闲的同一 GPU 上比较完整 forward 耗时：

```bash
python experiment/mathseg_uav/inference.py benchmark \
  --checkpoint runs/mathseg_uav/primary/M3_seed3407/best.pt \
  --size 1024 --warmup 20 --repeats 100
python experiment/mathseg_uav/run.py summarize --output-root runs/mathseg_uav/primary
```

测速包含先验、骨干、融合、双头和上采样；不含文件 I/O 或主机到 GPU 的数据传输。
固定 FP32、batch 1，无 TTA，CUDA 同步，报告 mean/median/P95、吞吐量和峰值 allocated 显存。
`--profile-flops` 是 PyTorch profiler 支持算子的**不完整** FLOPs 估计，不能当作总 FLOPs；
尚未实现覆盖所有算子的 FLOPs 计数。模型返回的尺度权重和先验图可用于后续解释性可视化。
汇总生成逐运行 `summary.csv`，以及只对完成且训练协议一致的运行聚合的 `aggregate.json`；
标准差使用样本标准差，单种子时为 null。汇总指标均为验证集分数。

## 本地回收日志

训练完成后从本地执行（替换主机、SSH 端口和服务器项目路径）：

```bash
mkdir -p experiment/mathseg_uav/logs/primary
rsync -av -e 'ssh -p PORT' \
  --include='*/' --include='*.log' --include='*.json' --include='*.jsonl' \
  --include='*.csv' --include='exit_code' --exclude='*' \
  USER@HOST:/SERVER/PROJECT/runs/mathseg_uav/primary/ \
  experiment/mathseg_uav/logs/primary/
```

只回收日志、配置、指标，不下载权重或预测。可在运行中重复回收快照；本脚本没有假装自动同步，
服务器地址和连接权限确认后才应配置实际同步任务。

## 轻量回归测试

从项目根目录执行：

```bash
python -m unittest experiment.mathseg_uav.test_mathseg -v
```

测试不下载预训练权重，使用微型随机 SegFormer 和临时合成图像；覆盖所有变体反向传播、
M2/M3 配对、先验边缘对齐、Ignore 损失/边界/指标、划分交集、类别权重、采样器恢复、
训练与恢复一致性、原图评估、预测导出和测速入口。不替代真实 B3/CUDA 的服务器短程检查。
