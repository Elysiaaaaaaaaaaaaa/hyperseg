# H2：Swin-L 骨干 + Mask2Former 解码器（赛方基线对照，20k iter）

状态：**已完成**。2026-09-30 21:12 在 server2 启动正式 20k，23:21 正常收官，
`best_mIoU_iter_20000.pth` 记录 **val mIoU 69.80%**；同日 23:48 完成复赛
`test_2`（1300 张）整图导出并通过两级格式自检。完整运行记录与对照分析见
[experiment.md](experiment.md)。配置为
[mask2former_swin_l_512.py](mask2former_swin_l_512.py)，启动脚本为
[launch_h2_server.sh](launch_h2_server.sh)，服务器侧守候为
[watch_h2_remote.sh](watch_h2_remote.sh)，导出驱动为
[run_infer_h2_test2.sh](run_infer_h2_test2.sh)。

启动前实测（20 iter 冒烟）：训练峰值显存 **8.4 GB**、**0.30 s/update**、
原生验证 699 张 **约 80 s**、检查点（关优化器状态）**827 MB**。
实际全程 2.15 h，与 ETA 一致。

**主要结论（与 09-30 big fix 五臂同预算对照）**：等预算（20k）下，
Mask2Former 解码 **0.6980** 未能追平仓库 HyperSeg 解码的 `baseline` **0.7038** /
`aug` **0.7042**，但高于三个假背景惩罚臂（0.6926 / 0.6863 / 0.6837）。
在 `test_2` 上，H2 在"已经塌陷的那批图"里把背景吃回去的能力明显更强
（H3 塌陷池 146 张：H3 给 Building 2.9%，H2 给 **22.8%**），
代价是非塌陷图背景占比整体上浮，全局背景占比反而最高（0.3566 vs 0.3419 / 0.3087）。
即**换了一种塌陷方式，而不是整体修好**。注意 H2 的 `test_2` 导出是**整图 1024**，
bgfix/H3 是 **512 滑窗**，跨协议只能看方向与排序。

## 1. 目的与对照

赛方公布的基线采用 Swin-L + Mask2Former。本实验（H2）按 20000 次迭代的
预算，在本仓库固定数据与评估协议下复现该架构，作为提交成绩的官方基线
参照，并补齐 H1/H3 的对照矩阵：

| 编号 | 骨干 | 解码路径 | 状态 |
| --- | --- | --- | --- |
| H0 | MiT-B3 | 仓库 HyperSeg | 原实现已有；需按 H3 文档协议重跑配对基线 |
| H1 | MiT-B3 | Mask2Former | 已完成 160k，验证 mIoU 76.51%（148400 iter） |
| H2 | Swin-L | Mask2Former | 本文新增，20k iter |
| H3 | Swin-L + 通道投影 | 仓库 HyperSeg | 已完成 160k，验证 mIoU 77.66%（137200 iter） |

H2 与 H1 构成同解码器（Mask2Former）下的骨干对照；H2 与 H3 构成同骨干
（Swin-L）下的解码器对照。H2 预算为 20k，与 H1/H3 的 160k 不同，跨预算
比较只能表述为“赛方基线预算下的架构表现”，不能当作等预算消融结论。

## 2. 架构与权重来源

骨干直接采用官方 MMSegmentation Swin-L 配置
（`mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py` 的
backbone 段）：patch4/window12/pretrain 384、embed_dims 192、
depths `[2,2,18,2]`、heads `[6,12,24,48]`、drop_path 0.3、
`patch_norm=True`，输出四级步长 `[4,8,16,32]`、
通道 `[192,384,768,1536]`。Mask2Former 的 pixel decoder、transformer
decoder 与损失（含 no-object 类权 0.1）继承 R50 基础配置，与 H1 一致。

**一处 API 适配（2026-09-30 冒烟抓到）**：本环境 MMSegmentation 1.2.2 的
`SwinTransformer.__init__` **不接受 `convert_weights`**（该参数属旧的
mmcls 风格转换开关，1.x 已移除）。官方 swin-b/l 配置同样不传它，
因此配置里已去掉；`pretrain_img_size` / `qk_scale` / `frozen_stages` 等
按官方 block 保留。传了会在 `MODELS.build(backbone)` 直接抛
`TypeError: unexpected keyword argument 'convert_weights'`，
不是加载失败而是构造失败。

- 骨干权重：<https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth>
- 服务器本地副本：`models/swin_l/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth`，
  SHA-256 `dc532f0ad8f98b71e0c88e8d673512b1526af9cba2cf0ab71130478423ffb3ff`
  （与 H3 启动时校验的同一文件）。
- 与 H3 相同的已知加载差异：官方分类预训练文件不含四级输出 norm，这些层
  默认初始化；其余骨干参数应全部加载。训练前必须在日志中核对缺失/多余键，
  不允许骨干大面积未加载而继续正式训练。
- 不加载任何 ADE20K 分割权重；Mask2Former 解码部分随机初始化。

## 3. 训练协议

除“骨干、迭代预算、验证频率”三处外，其余全部沿用 H1，保证协议可比：

| 项目 | H2 设置 | 备注 |
| --- | --- | --- |
| 数据 | 固定 train 5598 / val 699 / test 699，复用 `runs/splits` | 与 H1/H3 一致 |
| 标签 | 原始 0 为 Ignore；`reduce_zero_label` 后 8 类，ignore 255 | 与 H1 一致 |
| 训练裁剪 | 512×512 | 官方 ADE20K 配置为 640×640，此处为与 H1/H3 对齐并适配单卡 4090 的唯一协议性差异 |
| 数据增强 | RandomResize(1024, 0.5–1.5) + RandomCrop(cat_max_ratio 0.9) + 水平/垂直翻转 + PhotoMetricDistortion | 与 H1 一致 |
| 输入标准化 | ImageNet mean/std，BGR→RGB | 与 H1 一致 |
| Batch / 累积 / 精度 | 2 / 1 / FP32 | AMP 在本环境匈牙利匹配会产生非有限 cost，保持禁用（H1 已确认） |
| 优化器 | AdamW，LR 1e-4，骨干 LR 倍率 0.1，weight decay 0.05，clip max_norm 0.01 | 与 H1 一致；另按官方 Swin 配置对 `absolute_pos_embed`、`relative_position_bias_table` 关 weight decay |
| 学习率计划 | PolyLR，power 0.9，end=20000 | 随预算缩短 |
| 迭代预算 | 20000 iter（40000 样本视图，约 7.1 等效 epoch） | 赛方基线预算 |
| 验证频率 | 每 1000 iter，结束时额外验证；checkpoint 同频、保留最佳 mIoU | 20 次验证，便于选取最佳点 |
| 随机种子 | 3407 | 与 H1/H3 首轮一致 |
| 模型选择 | 验证集 mIoU 最佳；独立 test 不参与选模或调参 | 与 H1/H3 一致 |

如 batch 2 显存不足：先把配置中骨干 `with_cp` 改为 `True`（Swin gradient
checkpointing，H3 已用同法）；仍不足则按 README 惯例改 batch 1、累积 2、
迭代数翻倍为 40000、val_interval 2000，保证 optimizer update 数与样本
视图数不变。任何改动需同步记录到本节，不静默变更预算。

## 4. 评估与交付

- 验证与独立 test：原始分辨率整图（whole）推理、batch 1、无 TTA，
  IoUMetric ignore_index 255；与 H1 完全同协议。不使用 MMSeg 1.2.2 的
  slide 模式（Mask2FormerHead 按 `pad_shape` 还原尺寸的已知问题，见 README）。
- 产出命名：`runs/mask2former_uav/h2_swin_l_mask2former_20k_seed3407_fp32/`，
  含训练日志、配置转储、`best_mIoU_iter_*.pth`、`iter_20000.pth`。
- 训练结束后：独立 test 评估一次（仅报告，不参与调参）；对复赛
  `test_2/images`（1300 张）用 `mask2former_swin_l_512_submission.py`
  导出，并经 `tools/check_submission.py` 检查同名、单通道、1024×1024、
  像素 0..8。
- 日志与指标回收到本目录 `logs/`、`results/`；权重留服务器，不提交仓库。

## 5. 实现与启动前验收

1. 配置 [mask2former_swin_l_512.py](mask2former_swin_l_512.py) 与统一包装器
   `run.py train --model mask2former_swin_l`；`--backbone-checkpoint`
   同时注入 `HYPERSEG_SWIN_L_CHECKPOINT`。
2. 启动脚本 [launch_h2_server.sh](launch_h2_server.sh) 先断言
   `nvidia-smi` 有卡且 `torch.cuda.is_available()` 为真，再校验骨干权重
   SHA-256，最后经 `run.py` 启动；带 `flock` 防重复启动与 `exit_code` 收尾。
3. 正式训练前先做 smoke：`run.py train --model mask2former_swin_l
   --max-iters 20 --val-interval 20 --work-dir runs/mask2former_uav/h2_smoke`，
   确认 loss 有限、骨干加载日志无缺键异常、峰值显存可承受。
4. 通过后后台启动正式 20k 训练；训练结束按第 4 节评估、导出并回收日志，
   将分析写入本目录 `experiment.md`。

启动命令（服务器，后台）：

```bash
mkdir -p runs/mask2former_uav/h2_swin_l_mask2former_20k_seed3407_fp32
nohup bash experiment/mask2former_uav/launch_h2_server.sh \
  > runs/mask2former_uav/h2_swin_l_mask2former_20k_seed3407_fp32/train.log \
  2>&1 < /dev/null &
```

## 6. 结果与产物（2026-09-30）

### 训练

| 项目 | 值 |
| --- | --- |
| 最佳检查点 | `best_mIoU_iter_20000.pth`（827 MB，关优化器状态） |
| best val mIoU | **69.8000%**（aAcc 84.97 / mAcc 80.66），step 20000 |
| 平台区间 | 16000 → 68.45、18000 → 69.79、19000 → 69.42、20000 → 69.80（20k 已跑满） |
| 训练耗时 | 21:12 → 23:21，约 2.15 h |
| 工作目录 | `/root/hyperseg/runs/mask2former_uav/h2_swin_l_mask2former_20k_seed3407_fp32` |

标量读取：`<work_dir>/20260930_211218/vis_data/scalars.json`，**验证行用裸键 `mIoU`**
（本环境 MMSeg 1.2.2 不写 `val/mIoU`），判据是"该行含 `mIoU`"。

### test_2（复赛无标注，1300 张）导出

| 项目 | 值 |
| --- | --- |
| 协议 | MMSeg 整图 `mode='whole'`，原生 1024×1024，`size_divisor=32`，无 TTA / 无滑窗 |
| 耗时 / 峰值显存 | 213 s（0.131 s/张）/ 2020 MB |
| 本地目录 | `experiment/mask2former_uav/results/h2_swin_l_mask2former_20k_test2/pred_test2_h2_20k/` |
| 归档 | `pred_test2_h2_20k.tar.gz`，13864622 B，SHA-256 `3eff53a191b10ca0f26f38b7c97f6fdf342d5bca1178f69b505ac2fcb3c73888` |
| 自检 | 服务器 `OK: 1300 predictions`；本地复查 1300× PNG 颜色类型 0、8-bit、1024×1024、无 PLTE/tRNS、文件名与输入一一对应 |
| 对照产物 | `results/test2_compare_h2_20k.json`、`results/test2_montage_h2_20k.png` |

### 仍未做

- 固定有标注 `test.txt` 划分上的独立 `eval`（本次只做了 val 与 `test_2` 导出）。
- 更长预算的延长臂：当前证据只能支撑"20k 下 Mask2Former 解码不及 HyperSeg 解码",
  不能推断"该解码器本身更弱"。→ 见第 7 节（H2b：20k→60k 续训，代码已就绪）。
- 与 bgfix/H3 的 `test_2` 协议对齐（H2 整图 vs 其 512 滑窗）。

## 7. H2b：从 20k 检查点续训到 60k iter（2026-10-01 准备完毕）

目的：把"20k 是个平台还是没训够"这件事从猜测变成数据。H2 的 val mIoU 在
16000–20000 停在 68.45 / 69.79 / 69.42 / 69.80，只说明**在 20k 预算内**它没
追平仓库 HyperSeg 解码，不能说明解码器本身更弱。续训臂把这个不确定性定量化。

状态：脚本与配置就绪，**未启动**（server2 当时处于关机状态）。

### 7.1 为什么不能直接 `--resume`（本环境实测的陷阱）

MMSeg 1.2.2 的 `tools/train.py` 里 `--resume` 是 `store_true`，不带路径，只能
从 `<work_dir>/last_checkpoint` 指针自动取检查点；因此要指定文件必须走
`load_from` + `resume=True`（`Runner.load_or_resume`：`resume and load_from is
not None` 时直接用该路径）。更关键的是**参数调度器状态会被一并恢复**：

- `CheckpointHook.save_param_scheduler` 默认 `True`，**不受
  `save_optimizer=False` 影响**，所以 H2 的检查点里带 `param_schedulers`；
- `_ParamScheduler.state_dict()` 返回 `self.__dict__`（只排除 `optimizer`），
  含 `end=20000`、`last_step=19999`、`_global_step=19999`；
- `load_state_dict()` 是 `self.__dict__.update(...)`，**直接覆盖**配置里的
  `end=60000`/`end=40000`；
- 而 `_ParamScheduler.step()` 只在 `begin <= _global_step < end` 时写 LR ——
  `_global_step` 从 19999 起继续递增，`< end=20000` 很快不再成立，
  **此后 LR 再也不会被更新**，40k iter 会在一个冻结的 LR 上跑完。

即：天真续训不会报错、不会崩，只会安静地给出一个错误的结论。

### 7.2 LR 锚定与等价性证明

处理方式：用 `prepare_resume_checkpoint.py` 把 `param_schedulers`（以及万一存在的
`optimizer`）从检查点里剥掉，让配置新建的 PolyLR 生效，并把"新建计数器从 0 开始"
这件事用参数补偿回来。

MMSeg `PolyLR`（`eta_min=0`）的 `_get_value` 是乘性的，可望远镜化为闭式：

```
lr(n) = lr(0) · ((T − n) / T)^power,   T = end − begin − 1,  n = last_step（每 iter +1）
```

于是取 `end' = total − resumed`、初始 LR 取 `anchor = base_lr ·
((total−1−resumed)/(total−1))^power`，续训第 `n` 步的 LR 与"从头跑 total 次迭代的
运行在第 `resumed+n` 步"的 LR **逐点相等**（`(E−1)/T60 · (E−1−n)/(E−1) =
(E−1−n)/T60`）。本臂的取值：`base_lr=1e-4`、`power=0.9`、`resumed=20000`、
`total=60000` →

| 量 | 值 |
| --- | --- |
| `anchor`（= 60k 曲线在 iter 20000 的 LR） | **6.9424795567e-5** |
| `param_scheduler.0.end` | **40000**（新建计数器跨 0..39999） |
| 在 iter 60000 的 LR | 0 |
| 数值自检最大相对偏差 | 4.9e-16（`--self-test`，纯标准库） |

**必须一起读的诚实声明**：H2 的 20k 臂是用 `--no-save-optimizer` 存的，所以
Adam 的一阶/二阶矩无法恢复，续训开始时是"新优化器 + 已退火权重"。LR 曲线是精确
等价的，但自适应状态是重新估计的（几百步内收敛），且 LR 从 ~0 重新抬到
6.94e-5 —— 因此 **20k→60k 的增益里混着"更多迭代"和"LR 回热"两种效应**。
若要纯"更多迭代"的干净对照，只能从头跑 60k（约 6.3 h），本方案不含它。

### 7.3 协议

除下列各项外与 H2 完全相同（同骨干、同种子 3407、同 512 裁剪、同增强、同固定划分、
同 batch 2 / FP32 / AdamW 1e-4（骨干 0.1×）/ clip 0.01）：

| 项目 | H2b 设置 |
| --- | --- |
| 起点 | `h2_swin_l_mask2former_20k_seed3407_fp32/iter_20000.pth`（权重 + `meta['iter']=20000`） |
| 迭代预算 | 20000 → **60000**（续跑 40000 iter） |
| 学习率 | PolyLR power 0.9，`end=40000`，初始 6.9424795567e-5 |
| 验证频率 | 1000（4 万 iter / 40 次验证，与 20k 臂同频，续在同一条 iter 轴上） |
| 检查点 | 每 1000 iter，保留 2 个 + 最佳；磁盘 ≥12 GiB 时**保留优化器状态**（使本臂自身可续跑，因为它的 `end` 与配置一致、不存在 7.1 的过期问题） |
| 工作目录 | `runs/mask2former_uav/h2_swin_l_mask2former_20k_to_60k_seed3407_fp32` |
| ETA | 40000×0.30 s ≈ 3.3 h + 40×80 s ≈ 0.9 h ⇒ 约 **4.2 h** |

### 7.4 启动与闸门

```bash
setsid nohup bash experiment/mask2former_uav/launch_h2_server.sh resume60k \
    > runs/mask2former_uav/h2b_train.nohup.log 2>&1 < /dev/null &
setsid nohup bash experiment/mask2former_uav/watch_h2_remote.sh 300 \
    runs/mask2former_uav/h2_swin_l_mask2former_20k_to_60k_seed3407_fp32 h2b \
    runs/mask2former_uav/h2b_watch.log >/dev/null 2>&1 < /dev/null &
```

`resume60k` 模式在预检（CUDA/骨干哈希/数据划分）之后带三道闸，任一不过就
`kill` 整个进程组并写 `exit_code=4`：

1. **配置闸**（确定性、致命）：`<work_dir>/mask2former_swin_l_512.py` 是 MMSeg
   转储的**已解析**配置，必须同时含 `max_iters=60000`、`lr=6.9424795567e-05`、
   `end=40000`；
2. **进度闸**（确定性、致命）：日志必须出现 `resumed epoch: 0, iter: 20000`
   （`Runner.resume` 的原文），否则本次不是我们要的续训（例如 `load_from`
   未生效会从 iter 0 重跑）；
3. **LR 闸**（参考性）：`scalars.json` 第一条 `lr` 取各参数组最大值，应与 anchor
   相差 <2%，不符只告警不终止。

另外：源检查点缺失时**硬失败**（MMSeg 的 `--resume` 在空 work_dir 里会静默从头开始，
这是最危险的静默失败模式）；`--self-test` 每次启动都重跑一遍等价性数值校验。

### 7.5 读出方式

续训臂的 val 曲线与 20k 臂在**同一条 iter 轴**上（21000…60000），因此可以直接接在
20k 曲线后面看 20k / 40k / 60k 三个点，并与 bgfix 五臂的 20k 成绩并排。
收官后按第 4 节同样方式回收日志、做 `test_2` 导出。

## 8. H2-160k：从头训练到官方完整预算（2026-10-01 启动）

状态：**2026-10-01 00:20:57 已在 server2 启动**，工作目录
`runs/mask2former_uav/h2_swin_l_mask2former_160k_seed3407_fp32`，
启动方式 `bash launch_h2_server.sh long`。

### 8.1 它回答什么，以及不能拿它和 20k 臂怎么比

这个臂是**从零训练到 160000 iter**（官方 Mask2Former ADE20K 配置的完整预算），
用来回答"给足预算时这个解码器能达到什么水平"。

**关键：它的训练轨迹与 20k 臂不同，两者不可逐点对照。** PolyLR 的 `end` 是
160000，所以 iter 20000 时它的 LR 仍在 `1e-4·(140000/160000)^0.9 ≈ 8.9e-5`
（backbone 组 8.9e-6），而 20k 臂在 iter 20000 已经把 LR 退到 ~0。因此：

- 160k 臂曲线上 20000 点的 mIoU **不等于** 20k 臂的 69.80，两者不能相减；
- 想拿到"20k 臂的轨迹往后延长"的对照，必须走 7 节的 `resume60k`（重锚定 LR，
  与从头跑 60k 逐点等价），不能拿这个臂代替；
- 可与 20k 臂比的只有"最终水平"这个结论，并且要写明 LR schedule 不同。

### 8.2 协议

除预算与验证密度外，全部与 20k 臂一致（同骨干、同权重文件、同 seed 3407、
同 512 裁剪、batch 2×1、FP32、同固定划分、同增强）：

| 项 | 20k 臂 | H2-160k |
|---|---|---|
| `train_cfg.max_iters` | 20000 | **160000** |
| `param_scheduler.0.end` | 20000 | **160000**（PolyLR 在全程退火） |
| `train_cfg.val_interval` | 1000 | **2000**（2000 整除 20000，故 20k/40k/…/160k 均命中） |
| 检查点 | `--no-save-optimizer`，keep 2 | `--save-optimizer`，**keep 1**（约 2.6 GB/个） |

> 配置里 `param_scheduler` **只有一项**（PolyLR，无 warmup），所以
> `param_scheduler.0.end=<预算>` 改的就是它；实测 LR 逐点符合
> `1e-4·((160000−i)/160000)^0.9`（i=50 时 9.9972e-5）。

### 8.3 实测与 ETA

启动前冒烟闸照常通过（20 iter）。正式训练实测：

| 项 | 实测 |
|---|---|
| 速度 | **0.297–0.303 s/update**（≈0.300） |
| 显存 | **8.4 GiB**（8445 MB），GPU 利用率 **79%**，65 °C |
| grad_norm | 223–417（有限，无 NaN） |
| 验证 | 80 次 × 699 张原生 1024 ≈ 80 s ⇒ ≈ 1.8 h |

`160000 × 0.300 ≈ 13.3 h` + 80 次验证 ≈ 1.8 h + 检查点写入 ≈ 0.2 h
⇒ **ETA ≈ 15.3 h，预计 2026-10-01 15:30–16:00 收官**。
（MMSeg 自报的 `eta` 在早期只外推训练、未计入后续验证，会低估约 1.5 h。）

### 8.4 崩溃恢复

15 h 的运行跨过平台的"GPU 中途消失"风险窗口（见 2026-09-30 记录），所以本臂
**保留优化器状态**，并用同一个启动器恢复：

```bash
H2L_RESUME=1 bash experiment/mask2former_uav/launch_h2_server.sh long
```

这里用普通 `--resume` 是**安全**的：检查点里 scheduler 状态的 `end=160000`
与配置一致，不存在 7 节那种"旧 `end` 冻结 LR"的问题。脚本会校验
`last_checkpoint` 必须存在（缺失即 exit 2），并跳过冒烟闸。

磁盘：`--save-optimizer` 下 2 个检查点（最新 + best）约 5.2 GB，系统盘启动时
14 GiB 空闲，余量足够。

### 8.5 中途检查点测试（2026-10-01 09:16，训练未中断）

不等收官，用当时的 val 最优检查点 `best_mIoU_iter_82000.pth` 先测一轮。推理与训练
共卡并行（推理峰值 2.0 GB，训练 9.7 GB），训练未受影响。

| 测试 | 结果 |
|---|---|
| `test.txt` 独立评估（699 张有标注，选模型时未用） | **mIoU 77.65** / aAcc 88.82 / mAcc 86.53 |
| 同臂训练期 val（iter 82000） | 76.23 |
| `test_2` 1300 张整图导出 | exit 0，300 s，1300 张全部通过提交格式复核 |

**test(77.65) > val(76.23)**，差 +1.42 ⇒ 这一臂没有对 val 过拟合。
逐类最弱是 **Barren 53.02**（Background 69.53 次弱）。

同协议对照（H2 两臂都是整图 1024）的 val：20k 69.80 → 82k **76.23**，
比 bgfix baseline 的 70.38 高 **+5.85**。**之前"Mask2Former 解码不如 HyperSeg 解码"
的结论应改判为"20k 预算不够"**：预算的杠杆远大于调损失（bgfix 那一轮调损失只动了 ±1.5 点）。

`test_2` 跨协议画像与 Barren 类级坍塌风险见 [experiment.md](experiment.md) 的
"2026-10-01 09:16–09:24" 一节。复现命令：

```bash
bash experiment/mask2former_uav/run_h2_ckpt_test.sh <checkpoint> <label> [--skip-export]
```

收官后用同一条命令换 `<ckpt>` 即可测最终检查点；`--skip-export` 只跑有标注评估。

## 9. H2-160k 收官与 best 点的 test_2 导出（2026-10-01 16:16）

状态：**已完成**。160000 iter 跑满，`exit_code=0`，全程 15 h 35 m
（00:21:04 → 15:56:53），比 8.3 节的 ETA 15.3 h 略长。

| 项 | 值 |
| --- | --- |
| 最佳检查点 | `best_mIoU_iter_132000.pth`（911 MiB） |
| best val mIoU | **78.20**（step 132000） |
| 末点 val mIoU | 78.09（step 160000，aAcc 89.61 / mAcc 88.54） |
| `test.txt` 独立评估 | **mIoU 78.56** / aAcc 89.35 / mAcc 87.25 |
| `test_2` 导出 | 1300 张，214 s，本地格式复核 **PASS** |

**best 落在 132000 而不是末点，本身就是结论**：132000 之后 28 次验证都在
77.25–78.09 之间抖动，没有净增益 ⇒ 官方 160k 预算对这一臂是够用甚至偏长的。

三件需要一起读的事：

1. **`test.txt` 从 82k 的 77.65 涨到 78.56（+0.91）**，8 类里 7 类上涨，
   唯一下降的是 Barren（53.02 → 52.51），但它的 Acc 反升 +3.58。
2. **同一次训练在无标注 `test_2` 上把背景偏置加重了**：背景占比 0.3479 → **0.3649**
   （全部臂最高），h3 塌陷池里 0.8274 → 0.8581。20k 臂那种"把背景吃回去"的表现
   （塌陷池 0.7321 / Building 0.2281）反而退回了。
3. 因此**不能用 test.txt 的提升去推断 test_2 也变好**；两件事实不矛盾——
   延长迭代把训练先验拟合得更牢，而 `test_2` 的问题是分布外覆盖（先验背景仅 27.8%，
   模型在塌陷池里给到 86%），迭代次数不解决它。

完整逐类表、逐图净变化与对照蒙版见
[experiment.md](experiment.md) 的「2026-10-01 15:56–16:16」一节；
格式复核脚本 [check_h2_test2_predictions.py](check_h2_test2_predictions.py)。

