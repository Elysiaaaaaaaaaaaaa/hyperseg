# Few-shot 实验记录

本文档汇总本项目至今（2026-09-15）在 LoveDA 与 SUIM 上做过的全部 few-shot 实验，包含协议、配置、结果、结论与残留问题。所有数字来自本地已下载的日志与 `summary.json` / `results.csv`，未重新训练、未修改训练代码。

除特别说明外，指标均为 **mIoU 百分数**。

---

## 1. 实验的共同性质

**这些实验不是 episodic 元学习。** 全部是「固定支持集 + 跨数据集少样本监督微调」：从一个 UAV 预训练 checkpoint 初始化，在人工或规则挑出的少量完整标注图上做有监督微调，然后在固定评估集上测一次。

| 项 | LoveDA | SUIM |
|---|---|---|
| 源 checkpoint | `models/hyperseg_resume_best.pt`（v2，9 通道） | 同左 |
| 目标类别 | 8 通道，评估 1..7（0=nodata 忽略） | 8 通道，评估 0..7（BW 是有效类） |
| 支持集 | 每类别 K 张，跨类取并集 | 每类别 K 张，跨类取并集 |
| 评估集 | LoveDA Val 内划出的 1599 张（支持图已排除） | 官方 TEST 110 张，原尺寸 |
| 指标口径 | 7 类 mIoU | 8 类 mIoU |
| 训练默认 | AdamW lr=1e-4，batch=2，crop 512，poly 衰减，AMP，梯度裁剪 1.0，BN running stats 全冻结 | 同左（lr/crop/batch 为代码默认值，见 §6） |

两条重要口径限制：

- **LoveDA 结果是 Val 内部划分，不是官方完整 Val 指标。** 支持图来自 Val，且由 GUI 人工挑出，存在选择偏差；三个 seed 的标准差只反映训练随机性，不含支持集敏感性。
- **所有实验只在最终步评估一次**，没有训练中的验证曲线，因此不能定位过拟合发生在哪一步，也不能据此挑 epoch。

`fewshot_hyperseg/` 下的 episodic 模型（prototype /  episodic sampler）**从未参与任何一次实验**，与下述结果无关。

---

## 2. 实验索引

| # | 日期 | 数据集 | 内容 | 状态 |
|---|---|---|---|---|
| E0 | 09-10 | LoveDA | 旧协议 pilot：每域 5-shot | 完成（已被后续协议取代） |
| E1 | 09-12 10:11 | LoveDA | GUI 人工挑支持集并导出协议 | 完成（协议产物） |
| E2 | 09-12 14:37 | LoveDA | v2 主实验：0/1/2/5/10-shot × 3 seeds，adapter 2000 步 | 完成 13 次 |
| E3 | 09-12 15:03 | LoveDA | 单种子消融：adapter_200 / sh_2000 / sh_200，1/2-shot | 完成 6 次 |
| E4 | 09-13 14:59 | SUIM | GUI 人工挑支持集 | 完成，**未被采用** |
| E5 | 09-14 20:22 | LoveDA | 10-shot semantic-head 200 步 | 先失败后成功 |
| E6 | 09-14 20:25 | SUIM | 生成固定协议（seed 3407） | 完成 |
| E7 | 09-14 20:27 | SUIM | 4 setting × 4 shots × 1 seed | 先失败后成功，16 次 |
| E8 | 09-15 | SUIM | 结果诊断与下一轮建议 | 完成（分析报告） |
| E9 | 09-16 | UAV | Mask2Former + MiT-B3 对照实验配置与运行入口 | 已落地，待数据下载后运行 |

---

## 3. 逐次实验

### E0 · LoveDA 旧协议 pilot（09-10）

早期 `train.py`「每域 K 张」协议：Urban 和 Rural 各取 5 张，共 10 张训练图，源权重为 v1 `models/hyperseg_b3_best.pt`，50 epoch × 100 steps，评估 1659 张 Val 内部划分图，**按验证 mIoU 选最佳 checkpoint**。

- 结果：best mIoU **44.84**（background 19.63 / building 62.73 / road 56.37 / water 66.47 / barren 10.31 / forest 40.76 / agriculture 57.61）
- 该协议后续被「每类别 K 张 + 固定 1599 张评估集 + 最终步评估」取代，两者数字不可直接比较（评估集与模型选择方式都不同）。

产物：`runs/loveda_fewshot_valsplit/adapter_5shot_seed3407/`（`summary.json` 含混淆矩阵）

### E1 · LoveDA 支持集导出（09-12）

GUI 逐类人工挑选，导出嵌套支持集（每类有序列表取前 K 张）与共用评估清单。

| 每类 K | 去重后训练图 | 评估图 |
|---:|---:|---:|
| 0 | 0 | 1599 |
| 1 | 7 | 1599 |
| 2 | 14 | 1599 |
| 5 | 35 | 1599 |
| 10 | 70 | 1599 |

协议 SHA-256 `8184bef3…`，所有 K 共用同一评估集。产物：`runs/loveda_manual/export_20260912_101105_509451/`

### E2 · LoveDA v2 主实验（09-12）

源权重换成 v2 `hyperseg_resume_best.pt`，迁移时严格校验参数名/尺寸，8 通道头复制源通道 0..7、丢弃 Vehicle 通道 8。adapter 模式（scene / modulation / fusion / low-rank / decoder / boundary_refine / heads，893721 可训练参数，占 1.99%），2000 步，1024 整图评估无 TTA，最终步为结果。

| K | 支持图 | mIoU（3 seeds） | std |
|---:|---:|---:|---:|
| 0 | 0 | 53.34 | — |
| 1 | 7 | 49.33 | 0.44 |
| 2 | 14 | 51.59 | 0.09 |
| 5 | 35 | 53.30 | 0.27 |
| 10 | 70 | **55.36** | 0.27 |

逐类 IoU（3 seeds 均值，%）：

| K | backgr. | building | road | water | barren | forest | agric. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 55.00 | 64.88 | 56.31 | 70.50 | 29.03 | 37.58 | 60.07 |
| 1 | 53.83 | 62.47 | 57.12 | 63.10 | 31.52 | 30.31 | 46.96 |
| 2 | 54.21 | 59.17 | 56.53 | 65.81 | 34.46 | 35.23 | 55.72 |
| 5 | 54.27 | 64.79 | 57.53 | 67.30 | 34.27 | 37.84 | 57.11 |
| 10 | 52.87 | 66.17 | 56.43 | 67.90 | 41.48 | 42.35 | 60.34 |

产物：`runs/loveda_manual_v2_server2/completed_20260912/`（`results.csv`、`aggregate.json`、各 run 的 `history.jsonl` / `final.pt` / `summary.json`）

### E3 · LoveDA 单种子消融（09-12）

动机：E2 中 1-shot（49.33）反而低于 0-shot（53.34），怀疑 adapter 参数过多导致极少样本过拟合。固定 seed 3407、同一支持集与评估集，只改**更新范围**和**训练预算**，共 6 次。

| 设置 | 可训练参数 | 步数 | 1-shot | 2-shot |
|---|---:|---:|---:|---:|
| 0-shot 基线 | 不更新 | 0 | 53.34 | 53.34 |
| adapter（E2 基线） | 893721 | 2000 | 49.81 | 51.63 |
| adapter_200 | 893721 | 200 | 51.47 | 51.29 |
| semantic_head_2000 | 520 | 2000 | 52.54 | 51.38 |
| semantic_head_200 | 520 | 200 | **53.59** | **53.53** |

要点：

- 仅语义头 + 200 步在两个 K 上都是本轮最高，相对 0-shot 只提高 0.25 / 0.19 个百分点。
- 单独缩短 adapter 步数、或单独缩小到语义头（仍 2000 步），都没能在两个 K 上一致获益 —— 不支持「结构太复杂是唯一原因」。
- 520 参数下 2000 步仍比 200 步差 1.05 / 2.15 个百分点，说明低参数量也挡不住过度适应，**训练预算不可忽略**。
- 语义头 1-shot：200 步组前/末 20 步平均 loss 1.3649/1.1644，2000 步组 1.3647/0.6789 —— 拟合支持集更好但评估更差。
- 200 步与 2000 步使用各自总周期的学习率衰减，短预算不是长轨迹的前缀。

产物：`runs/loveda_ablation_v2_server2/completed/`（`analysis.md`、`ablation_results.csv`、6 个 run 目录）

### E4 · SUIM 人工支持集（09-13，**未采用**）

GUI 导出 `runs/suim_manual/export_20260913_145956_961191`，10-shot 为 78 张，协议哈希 `16da76ae…`。服务器上真正使用的是 E6 自动生成的协议（10-shot 79 张，哈希 `b8f2e011…`）。**两套不同，不能把 E7 的结果称作该 GUI 协议的结果。**

### E5 · LoveDA 10-shot semantic-head 200 步（09-14）

用 E3 的最优设置去检验 10-shot，判断「少更新范围」能否推广。

- 结果：mIoU **53.59**（同 seed 3407）
- 对照：10-shot adapter-2000 同 seed 为 **55.47**，0-shot 为 53.34。
- 即：10-shot 上 semantic-head 只比 0-shot 高 0.25，比 adapter 低 **1.88** 个百分点 —— **E3 的结论不能推广到大 K**。

过程：首次提交在 20:22 失败（见 §6），修复后 20:24 跑通。

产物：`runs/loveda_latest_10shot_semantic_head_200_seed3407/`，日志 `logs/fewshot_20260914/loveda_10shot_semantic_head_200_seed3407.log`

### E6 · SUIM 固定协议生成（09-14）

扫描 1525 张训练 mask 做审计，按「目标类 ≥512 像素且占比 ≥0.05%」自动挑选，seed 3407。

| K | 1 | 2 | 5 | 10 |
|---:|---:|---:|---:|---:|
| 支持图 | 8 | 16 | 40 | 79 |

评估为官方 TEST 110 张，协议哈希 `b8f2e011…`。日志 `logs/fewshot_20260914/suim_prepare_protocol_seed3407.log`

### E7 · SUIM 实验矩阵（09-14）

4 种「更新范围 × 预算」× 4 个 K × seed 3407，共 16 次；随机初始化 8 通道语义头，无 0-shot。

| 更新范围 / steps | 1-shot | 2-shot | 5-shot | 10-shot |
|---|---:|---:|---:|---:|
| adapter / 200 | 20.27 | 21.72 | 20.34 | 18.41 |
| adapter / 2000 | **27.13** | **33.40** | **40.26** | **43.56** |
| semantic-head / 200 | 2.09 | 2.20 | 2.29 | 2.39 |
| semantic-head / 2000 | 16.56 | **17.48** | 16.06 | 16.26 |

- adapter 从 200 → 2000 步分别提高 6.86 / 11.68 / 19.92 / 25.15 个百分点；200 步时「shot 越多反而越差」应归因于预算不足，而非数据无益。
- 最佳设置（adapter / 2000 / 10-shot）逐类 IoU：BW 82.22、HD 49.62、PF **10.12**、WR 32.52、RO **19.29**、RI 55.66、FV 47.95、SR 51.10。**PF（植物/海草）和 RO（机器人/仪器）是主要弱类。**
- 训练 loss 末段均值 0.097–0.259，远低于测试表现，少样本拟合 ≠ 泛化。

日志：`logs/fewshot_20260914/suim_grid_seed3407_retry1.log`（以 `ALL_DONE` 结束），结果 `runs/suim_fewshot_v1_seed3407/`

### E8 · SUIM 诊断（09-15）

产出 `runs/suim_analysis_20260915/analysis.md` + `overview.png` / `overview.pdf`，给出按收益排序的优化建议：先修 head 学习率（head 与迁移模块共用 1e-4、且从第 1 步就开始衰减，可能训练不足），再查训练/推理尺度差异（训练随机放大 1.07–1.5 倍，测试原尺寸），然后改进弱类裁剪，最后才考虑解冻 encoder 最后阶段。

### E9 · UAV Mask2Former 对照实验（09-16，待运行）

在 `experiment/mask2former_uav/` 落地 MMSegmentation 实验，主实验为
Mask2Former + MiT-B3，控制组为 SegFormerHead + MiT-B3。二者固定使用同一
数据划分、MiT-B3 初始化、512 裁剪、优化器和 160k iteration 预算；原始标签
0 映射为 255 Ignore，1..8 映射为内部 0..7。验证和测试采用原分辨率整图
推理，因此不能直接与旧版 768×768 resize 指标比较。

已从 `splits.zip` 恢复固定划分到 `runs/splits/`：train 5598、val 699、test
699。当前数据集仍在下载，尚未执行模型构建、显存 smoke test 或训练，不能将
本项记为已有实验结果。运行说明与服务器命令见
`experiment/mask2former_uav/README.md`。

---

## 4. 结果总表

**LoveDA（7 类 mIoU %，1599 张 Val 内部评估集）**

| 设置 | 1-shot | 2-shot | 5-shot | 10-shot |
|---|---:|---:|---:|---:|
| 0-shot 迁移基线 | 53.34 | 53.34 | 53.34 | 53.34 |
| adapter / 2000（3 seeds 均值） | 49.33 | 51.59 | 53.30 | **55.36** |
| adapter / 200（seed 3407） | 51.47 | 51.29 | — | — |
| semantic-head / 2000（seed 3407） | 52.54 | 51.38 | — | — |
| semantic-head / 200（seed 3407） | **53.59** | **53.53** | — | 53.59 |

**SUIM（8 类 mIoU %，官方 TEST 110 张，seed 3407）**

| 设置 | 1-shot | 2-shot | 5-shot | 10-shot |
|---|---:|---:|---:|---:|
| adapter / 200 | 20.27 | 21.72 | 20.34 | 18.41 |
| adapter / 2000 | **27.13** | **33.40** | **40.26** | **43.56** |
| semantic-head / 200 | 2.09 | 2.20 | 2.29 | 2.39 |
| semantic-head / 2000 | 16.56 | **17.48** | 16.06 | 16.26 |

---

## 5. 结论

1. **最优更新范围依赖 K。** 1/2-shot 上限制更新（仅语义头 + 短预算）最好；5/10-shot 上 adapter + 大预算明显更优（LoveDA 10-shot 差 1.88，SUIM 10-shot 差 27.3）。不要把一个 K 上的结论外推。
2. **1-shot adapter 会掉到 0-shot 以下**（LoveDA 49.33 vs 53.34），极少样本时大范围微调有负收益风险。
3. **训练预算与更新范围必须一起看**，单独改任何一个都不产生一致收益；且不同预算对应不同学习率衰减周期，不能互相截取比较。
4. **跨域差距远大于 K 的影响。** SUIM 即使在 10-shot / 2000 步也只有 43.56，比 LoveDA 同设置低十余个百分点；航拍 → 水下的域差异是主矛盾。
5. **当前证据不支持「模型没学到东西」**：逐类 IoU 变化有类别倾向，训练 loss 也确实下降；问题是泛化不足 + 弱类（PF/RO）覆盖不足。
6. **统计强度有限**：LoveDA 主实验有 3 个训练 seed，但支持集固定；SUIM 只有 1 个 seed；所有结论都不宜直接宣称显著。

---

## 6. 踩过的坑

1. **`run_manual.py` 不接受 `--modes semantic-head`**（旧 choices 为 head/adapter/full），E5 首次提交 20:22 失败，日志 `loveda_10shot_semantic_head_200_seed3407_preflight_failed.log`；脚本补齐 `semantic-head` 后 20:24 重跑成功。
2. **SUIM `run_grid.py` 首次入口报 `ModuleNotFoundError: No module named 'experiment'`**（以脚本路径方式运行、项目根目录不在 `sys.path`），日志 `suim_grid_seed3407_entry_failed.log`；改用从项目根运行后成功（`suim_grid_seed3407_retry1.log`）。
3. **SUIM 协议不一致**：服务器用自动生成协议（10-shot 79 张），本地 GUI 导出是 78 张、哈希不同。若要改用人工集，必须单列「支持集选择」对照。
4. **下载包缺文件**：SUIM 每次运行的 `run_config.json`、`selection.json`、`history.jsonl` 以及协议目录的 `dataset_audit.json` 都没下载下来，因此 lr=1e-4、crop=512、batch=2 这些是**代码默认值推断**，不是从运行配置确认的。
5. **评估口径**：LoveDA 的 44.84（E0）与 53+（E2 起）不可比，前者是旧协议 + 按验证集挑最佳 checkpoint。

---

## 7. 缺口与下一步

- **SUIM 缺 seed 3408/3409 与 5/10-shot 的消融；LoveDA 消融只做了 1/2-shot。**
- 没有任何实验跑过 `full` 模式，也没有解冻 encoder 的实验（域差异假设未验证）。
- 缺独立开发验证集，所有超参数决策都直接或间接使用了同一评估集。
- `fewshot_hyperseg/` 的 episodic 模型代码存在但零实验。

建议的下一轮最小矩阵（详见 `runs/suim_analysis_20260915/analysis.md`）：先补齐配置文件、锁定支持集与开发集协议，以 5-shot / seed 3407 / 2000 步做定位，逐项验证 A0（复现基线 + 记录开发曲线）、A1（head lr=1e-3）、H1/H2（semantic-head lr 3e-4 / 1e-3）、S1（原尺寸 vs 短边 512 推理）、A2（弱类有效面积裁剪）、A3（解冻 encoder 最后阶段 lr=1e-5）；定稿后再铺满 shots × seeds。

---

## 8. 文件索引

**日志**

- `logs/fewshot_20260914/`：09-14 服务器队列全部日志（含两次失败记录、`queue.status`、`queue_console*.log`）
- `experiment/loveda_fewshot/logs/loveda_10shot_semantic_head_200_seed3407.log`
- `experiment/fewShot_SUIM/logs/suim_grid_seed3407_retry1.log`、`suim_prepare_protocol_seed3407.log`
- `runs/loveda_manual_v2_server2/completed_20260912/logs/manual_v2_0_1_2_5_10shot.log`
- `runs/loveda_ablation_v2_server2/completed/logs/ablation_console.log`
- `runs/loveda_fewshot_valsplit/adapter_5shot_seed3407.log`

**结果**

- `runs/fewshot_miou_20260914.md` / `.csv`：09-14 汇总（mIoU 为 0–1）
- `runs/loveda_manual_v2_server2/completed_20260912/runs/loveda_manual_v2/{results.csv,aggregate.json}`
- `runs/loveda_ablation_v2_server2/completed/{analysis.md,runs/loveda_ablation_v2/ablation_results.csv}`
- `runs/loveda_latest_10shot_semantic_head_200_seed3407/`
- `runs/suim_fewshot_v1_seed3407/{results.csv,aggregate.json}`
- `runs/suim_analysis_20260915/analysis.md`

**协议 / 代码**

- `runs/loveda_manual/export_20260912_101105_509451/`、`runs/suim_manual/export_20260913_145956_961191/`
- `experiment/loveda_fewshot/`（train.py 旧协议、run_manual.py 主入口、run_ablation.py 消融）、`experiment/fewShot_SUIM/`（prepare_protocol.py、run_grid.py）
- 说明文档：`experiment/loveda_fewshot/README.md`、`MANUAL_V2.md`、`experiment/fewShot_SUIM/README.md`
