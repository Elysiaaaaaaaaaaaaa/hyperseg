# v2 resume 检查点 · 复赛测试集 2 导出：运行与分析

对应任务：用 `models/hyperseg_resume_best.pt` 在 `low_altitude_2026/test_2/images`
（1300 张）上导出一份预测并回收。日期 2026-09-30，机器 server2（RTX 4090）。

---

## 1. 运行时间线

| 步骤 | 内容 | 结果 |
| --- | --- | --- |
| 1 | 读检查点元信息 | 730 张量、epoch 134、`version=v2`、记录 val 0.7631595 |
| 2 | 上传权重 540 532 365 B 到 server2 | 65 s，7.92 MiB/s，双向 SHA-256 一致 |
| 3 | 预检（原计划用 v1 链路） | **失败**：`ImportError: cannot import name 'translate_legacy_keys'` |
| 4 | 同步 09-29 键名修复，重试 | **失败**：`TypeError: unexpected keyword argument 'version'` |
| 5 | 认定 v2 架构，改用 `tools/model_1.py` | **失败**：`ModuleNotFoundError: No module named 'tools.model_1'` |
| 6 | 上传 `tools/model_1.py`（首次误传，见 §4） | 加载成功：730/730 张量、628 键转写 |
| 7 | 协议标定（分辨率 → 归一化 → 口径 → batch/TTA） | 命中：size 768 + batch 8，差 9.28e-07 |
| 8 | 导出 1300 张 | 201.3 s，`submission_check=passed`，`exit_code=0` |
| 9 | 下载与本地复核 | 哈希一致、手册复核 PASS |

---

## 2. 权重身份：这是一个 v2 检查点，不是 v1 的续训

`hyperseg_resume_best.pt` 名字像"v1 的续训版"，但参数集不同，必须换架构：

| 前缀 | v1 (`hyperseg_uav/model.py`) | v2 (`tools/model_1.py`) |
| --- | --- | --- |
| `encoder` | 628 | 628 |
| `fusion` | 12 | 40 |
| `low_rank` | 2 | 6 |
| `boundary_refine` | — | 12 |
| 合计 | 686 | 730 |

`model_config` 带 `"version": "v2"`，v2 构造器不接受该参数；先 `pop` 再构造。
正确加载方式（逐键映射 + 双向严格）见 [v2_model.py](v2_model.py)，与
`experiment/loveda_fewshot/run_manual.py::load_v2` 的做法一致 —— 说明这份权重在
LoveDA / SUIM 的 few-shot 实验里就是当 v2 源权重用的，只是**从没走过 9 类 UAV 的
test 集导出链路**。

因此 `tools/infer_hyperseg.py`、`tools/validate_hyperseg.py` 都不能用：
前者构造 v1 模型，后者虽然在 `--checkpoint` 默认值里指向本权重，却 `import` v1 的
`HyperSegUAV`，实际跑不通（属于遗留不一致，建议后续修掉或改名）。

---

## 3. 协议标定：768 / 0.5 / 无 TTA

仓库里没有 v2 的 UAV 训练脚本（server2 系统盘、server2 数据盘 `/root/autodl-tmp`
全量搜过 `boundary_refine` / `model_1`，只有 few-shot、MathSeg 与本次新建的文件），
所以分辨率无从查证，只能实测。四步收敛：

**(a) 分辨率网格** `v2_val_replay.py`（`logs/v2_resolution_replay.json`）

| size | 重放 Val mIoU | 与 0.7631595 之差 |
| --- | --- | --- |
| 768 | 0.740312 | 2.28e-02 |
| 512 | 0.704259 | 5.89e-02 |
| 1024 | 0.742605 | 2.06e-02 |

都不匹配。注意 v2 在 768 下（0.7403）**低于** v1 的 0.7420 —— 一个"改进版"不可能
更差，所以差异不可能只来自分辨率，协议里还有别的变量。

**(b) 归一化** `v2_val_replay.py --normalize imagenet`（`logs/v2_norm_grid.log`）
768 → 0.718650、1024 → 0.716095、512 → 0.677203，全面**变差**。
结论：v2 与 v1 相同，只做 `/255`，**不加** ImageNet 归一化。

**(c) 指标口径** `v2_metric_probe.py`（`logs/v2_metric_probe.json`）
一次前向里同时算四种口径：

| 口径 | size 768 | size 1024 |
| --- | --- | --- |
| 逐 batch 逐类 IoU 平均（v1 口径） | 0.740312 | 0.742605 |
| 全数据集混淆矩阵逐类平均 | 0.794108 | 0.794501 |
| 逐图逐类 IoU 平均 | 0.719472 | 0.721698 |

记录值 0.7631595 卡在 0.7426 与 0.7945 **之间**。逐图口径明显不对；剩下两个都不中。

**(d) batch 大小 × TTA** `v2_protocol_search.py`（`logs/v2_protocol_search.json`）
上一步给出的区间其实指明了方向：v1 口径是"batch 内逐类 IoU 再对 batch 平均"，而
batch 越大、越容易同时包含全部类别，该口径就越向全数据集口径靠拢。所以固定 size 768、
一次采集逐图混淆矩阵，离线扫 batch ∈ {1,2,4,8,16,…,699}，同时对三种变体
（plain / hflip / hflip-logits-TTA）各算一遍：

| 变体 | batch | 重放 Val mIoU | 与记录之差 |
| --- | --- | --- | --- |
| **plain** | **8** | **0.7631586** | **9.28e-07** ✅ |
| plain | 2 | 0.740312 | 2.28e-02 |
| plain | 4 / 16 | 未命中 | — |
| hflip（只翻不算） | 8 | 0.762869 | 2.90e-04 |
| plain | 699（= 全局） | 0.794108 | 3.09e-02 |

**结论：v2 的评估协议是 size 768、batch 8、无 TTA、只做 `/255`。** 复现到 1e-6，
既确认了权重被完整正确还原（730 个张量、628 条键名转换都对），也确认了导出协议
（768 / 0.5 重叠 / 无 TTA）与既有 v1 提交一致，两份预测可直接比较。

> 若当初照搬 v1 的 batch 2 去"复核"，会得到 0.7403 而误判成"加载有问题"。
> 记录在案的 `val_miou` 必须连口径一起复现，这一点值得写进流程。

---

## 4. 踩到的坑

1. **server2 代码是 09-26 快照**，缺 09-29 的键名修复 → `translate_legacy_keys` 不存在。
   同步 `hyperseg_uav/{__init__,model}.py` 与 `tools/infer_hyperseg.py`（原文件已备份到
   `/root/autodl-tmp/backup_b3_resume_20260930/`）。
2. **误把 v2 当成 v1** → `unexpected keyword argument 'version'`。见 §2。
3. **`tools/model_1.py` 服务器上没有** → `No module named 'tools.model_1'`。
4. **Git Bash 改写裸路径参数**：`put --remote /root/hyperseg/tools/model_1.py` 少写
   `MSYS_NO_PATHCONV=1`，结果文件被传到
   `/root/hyperseg/C:/Users/86189/.workbuddy/binaries/PortableGit/…/root/hyperseg/tools/model_1.py`。
   已 `rm -rf '/root/hyperseg/C:'` 清掉并重传；`find /root/hyperseg -maxdepth 3 -name 'C:*'`
   复查为空。**凡是把 `/root/...` 当独立参数传的命令一律要加 `MSYS_NO_PATHCONV=1`。**
5. `infer_v2_test2.py` 是内联推理，不产出 `infer.log`（那是 `run_inference.py` 的子进程
   日志），批量 `pull` 时不要把它列进去，否则 `FileNotFoundError` 中断整批下载。

---

## 5. 结果

- 预测包 `hyperseg_b3_resume_test2_predictions.zip`，12 085 143 B，
  SHA-256 `07d3b396ad45c11bb8d587d05823a35944e7d3bf9868998d832c92ce2db31e1e`（与服务器一致）。
- `check_submission.py` 通过；手册第十三节复核 **PASS**：1300 条目、CRC ok、
  无重复、无嵌套路径、全部 IHDR 颜色类型 0、文件名与官方测试集 2 逐一对应。
- 耗时 201.3 s（≈6.5 张/秒），显存与稳定性无异常（只有已知的 cuDNN
  `CUDNN_STATUS_NOT_SUPPORTED` 告警）。

### 与 v1 提交的差异（`logs/v1_v2_comparison.json`）

| 类别 | v1 份额 | v2 份额 | v2/v1 |
| --- | --- | --- | --- |
| Background | 33.9424% | 33.8287% | 0.997 |
| Building | 18.5237% | 18.0421% | 0.974 |
| Road | 6.8922% | 6.8885% | 0.999 |
| Water | 7.3448% | 7.7420% | 1.054 |
| Barren | 1.4527% | 1.0115% | **0.696** |
| Vegetation | 20.9443% | 21.0656% | 1.006 |
| Agricultural | 10.3020% | 10.8215% | 1.050 |
| Vehicle | 0.5978% | 0.5999% | 1.004 |

- 8.8992% 的像素被改判，1295/1300 张图有变化；变化最大的 `test2_1010.png` 达 83.2%。
- 主导变化是 **Barren 类被大幅回收**（降到 0.696 倍），Water 与 Agricultural 小幅增加。
  这与 v2 在验证集上高 2.1 个点的方向一致，但**无标注测试集上无法证实这是改善还是
  偏移**：`test_2` 没有真值，任何"更好"的说法都缺证据。

---

## 6. 残留问题与后续

1. **v2 的 UAV 训练脚本已丢失**（本地与两台服务器都搜不到）。目前协议靠反推标定，
   能复现 val 值，但训练侧的超参（是否 AMP、是否带 boundary 损失权重、学习率日程）
   无从核对。建议把该脚本补回仓库，否则下次换 checkpoint 还得重新标定。
2. `tools/validate_hyperseg.py` 的默认 `--checkpoint` 指向 v2 权重却用 v1 模型加载，
   属于必然报错的遗留配置；应改用 `tools/model_1.py` 或明确标注只支持 v1。
3. `hyperseg_uav/model.py`（v1）与 `tools/model_1.py`（v2）并存且同名类，容易误用。
   建议给 v2 换个明确的名字（如 `HyperSegUAVV2`）或在文件头写清适用范围。
4. 本次只是导出与格式交付，**不要**把这个包当作成绩更好的提交直接用：
   需要先在有标注代理集上验证 v2 相对 v1 的真实提升，再决定替换。
5. server2 系统盘 30 G 已用 13 G、数据盘 50 G 已用 48 G（剩 2.8 G）。
   本次上传的 540 MB 权重在系统盘 `models/`，如无后续用途可删（数据盘上
   `/root/autodl-tmp/hyperseg_manual_v2_20260912/models/` 与
   `hyperseg_ablation_v2_20260912/models/` 各有一份副本）。
