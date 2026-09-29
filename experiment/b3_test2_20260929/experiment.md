# HyperSeg-B3 复赛测试集 2 推理研究日志

本文件记录 `experiment/b3_test2_20260929/` 任务的研究过程与日志分析。
路径、协议和运行命令见 [README.md](README.md)。

## 2026-09-29：server3 上用 HyperSeg-B3 导出复赛预测并回收

**状态：1300 张预测全部完成，提交格式检查通过，ZIP、日志与校验记录已下载到本地。**

### 目标与前置条件核对

- 目标：在 server3 上用 `models/hyperseg_b3_best.pt` 对复赛无标注测试集
  `dataset/low_altitude_2026/test_2/images` 推理，把结果下载回仓库。
- server3 为 AutoDL 容器，RTX 4090 24 GiB，系统盘剩余 11 GiB、数据盘剩余 38 GiB。
  `/root/hyperseg` 已含项目代码、`.venv-mask2former`（PyTorch 2.3.0+cu121、Transformers 4.57.6）
  与完整数据集，无需重新上传数据。
- 逐项哈希核对：检查点 `models/hyperseg_b3_best.pt`
  `fb124302bafaec18aaa3f695dc021efead7bd00f710348d040888bb270148968`、
  `tools/infer_hyperseg.py`、`hyperseg_uav/*.py` 与本地完全一致；
  数据集含 500 张初赛图像、1300 张复赛图像、6996 对训练图像/掩码，
  固定划分为 train/val/test = 5598/699/699。

### 发现并修复：旧权重编码器键名不兼容

首次预检在 `load_state_dict` 处失败：检查点含 628 个
`encoder.backbone.stages.{i}.blocks.{j}...` 键（Transformers 5 命名），
而当前环境 Transformers 4.57.6 的 `SegformerModel` 期望
`encoder.backbone.encoder.block.{i}.{j}...`。仓库中
`experiment/loveda_fewshot/train.py` 与 `experiment/fewShot_SUIM/run.py` 里已各有一份
`legacy_encoder_key` 转换函数，说明该检查点此前只在少数实验脚本中被正确加载。

处理方式：

- 把转换收拢到 `hyperseg_uav/model.py`，新增
  `legacy_encoder_key(key)` 与 `translate_legacy_keys(state)`，并在包 `__init__` 导出；
- `tools/infer_hyperseg.py` 改为先转换再 `load_state_dict`（严格加载），并打印转换条数；
- `tools/test_hyperseg.py` 同步转换。该脚本原先用 `strict=False` 加载，
  缺 628 个骨干键时只打印一行统计而不报错，会静默输出无效结果；本次已修，
  但 `strict=False` 语义保留，后续若要更严格可再改成严格加载。

### 权重正确性对照复核

转换是否等价不能只看键数量。用 `val_sanity.py` 在与训练相同的验证协议下重放评估：
固定 `runs/splits/val.txt` 的 699 张、768×768、batch 2、
`hyperseg_uav.losses.mean_iou` 逐 batch 求平均后再平均（与 `train_hyperseg.py` 一致）。

| 项 | 值 |
| --- | --- |
| 检查点记录 val_miou | 0.7420226665631505 |
| 重放 val_miou | 0.7420232149377345 |
| 绝对差 | 5.48e-07 |
| 转换键数 | 628 |
| 耗时 | 28.76 s（含模型加载与数据读取） |

差异只有浮点噪声量级，说明转换后的权重与训练时完全等价，可以放心用于导出。

### 运行与日志分析

- 预检：输入 1300 张全部为 1024×1024、无重名；检查点 9 类、严格加载通过；
  `--check-only` 不执行推理。
- 推理：`launch_server3.sh` 以 nohup 后台启动，PID=2776，
  采用 FP32、768×768 滑窗、50% 重叠、无 TTA（与训练/验证分辨率一致）。
- `runner.log` 每 30 秒记录一次进度，依次为 172、375、580、786、993、1200，
  即稳定在约 200 张/30 秒（≈6.7 张/秒，每张 4 个 768×768 窗口）。
- 总耗时 224.83 秒（约 3 分 45 秒，含预检、推理、提交检查与打包，不含下载）；
  `exit_code=0`，`result.json` 标记 `status=complete`、`submission_check=passed`。
- `infer.log` 中有一条 cuDNN `Plan failed ... CUDNN_STATUS_NOT_SUPPORTED` 警告，
  是 cuDNN 在 fp32 某卷积配置上回退到其他实现，属常见告警，推理继续正常完成；
  同时确认日志首行输出 `translated 628 legacy Transformers-5 encoder keys`。
- `tools/check_submission.py` 输出 `OK: 1300 predictions`：预测与输入同名、
  单通道 L、1024×1024、像素值 0..8。

### 结果包与本地复核

- 本地提交文件：[hyperseg_b3_test2_predictions.zip](hyperseg_b3_test2_predictions.zip)，
  12 655 462 字节（≈12.07 MiB），ZIP 根目录直接存放 1300 个预测 PNG。
- 服务器 ZIP 与本地重算 SHA-256 一致：
  `ff0a2fb0e0580bafb1b7f41f82d29861ad42bf181689f7efff43df3e84b0ef8a`。
- 本地逐张完整解码复核（`verify_download.py` → `logs/download_verification.json`）：
  ZIP CRC 通过、1300 个条目无重复、无嵌套目录；全部为 `L` 模式 1024×1024；
  类别取值集合为 {1..8}，未出现 0（模型未预测 Ignore）；
  文件名与本地 `test_2.zip` 的 1300 个成员完全一致。
- 预测类别占比：背景 33.94%、建筑 18.52%、植被 20.94%、农用地 10.30%、
  道路 6.89%、水体 7.34%、裸地 1.45%、车辆 0.60%，分布符合航拍场景先验。

### 口径与限制

- 复赛测试集只有图像、没有真值，**本次没有产生新的 mIoU 或比赛得分**；
  文中 74.2023% 只属于固定有标注验证划分，与复赛成绩无关。
- 本次未启用 TTA、未做多尺度融合或多模型集成，是单一检查点的直接导出。
  若后续要做 TTA 对比，可在同一输入上追加 `--tta` 重跑并单独存档，避免覆盖本结果。
- 服务器预测目录 `/root/hyperseg/runs/b3_test2_20260929/predictions` 保留；
  重跑前需先清理该目录，否则启动器会拒绝启动。

日志与依据：`logs/` 下的 [runner.log](logs/runner.log)、[infer.log](logs/infer.log)、
[submission_check.log](logs/submission_check.log)、[result.json](logs/result.json)、
[exit_code](logs/exit_code)、[preflight.json](logs/preflight.json)、
[val_sanity.json](logs/val_sanity.json)、
[download_verification.json](logs/download_verification.json)。

## 2026-09-29：按比赛手册第十三节逐条复核提交包

依据 [doc/无人机低空航拍图像语义分割_纯文本.md](../../doc/无人机低空航拍图像语义分割_纯文本.md)
第十三节（复赛输出格式与初赛相同），用 `check_manual_compliance.py` 直接解析 PNG 块结构复核，
结果见 [manual_compliance_20260929.json](logs/manual_compliance_20260929.json)，判定 **PASS**。

| 手册要求 | 复核结果 |
| --- | --- |
| 预测文件与测试集文件名完全一致 | 1300 个条目与本地 `test_2.zip`（SHA-256 `c8784dba…`，官方测试集 2）成员集合完全相同，无缺失、无多余 |
| 像素值严格对应类别 ID 0-8 | 全部文件取值集合为 {1..8}，落在 0..8 内；未出现 0（模型未预测 Ignore，合规） |
| 必须单通道 PNG，严禁 RGB 与调色板索引色 | 1300 个文件的 IHDR 颜色类型全为 0（灰度）；无 PLTE、无 tRNS 块；位深全为 8 |
| 尺寸与原始 1024×1024 严格一致 | 1300 个文件 IHDR 与 Pillow 读取均为 1024×1024，无缩放或裁剪 |
| 压缩为一个 zip 提交 | ZIP CRC 通过，1300 个条目，无重复条目、无嵌套目录、无目录项，非 PNG 条目为 0 |

补充事实：所有 1300 个 PNG 的块组成均为且仅为 `IHDR / IDAT / IEND`，是最简灰度 PNG，
不存在调色板、透明通道、交错或其他附加块；ZIP 内文件名与输入同名、
ZIP SHA-256 为 `ff0a2fb0e0580bafb1b7f41f82d29861ad42bf181689f7efff43df3e84b0ef8a`。

手册未对提交 ZIP 的文件名、ZIP 条目目录结构作额外硬性要求（仅要求压缩为一个 zip），
本次预测图直接放在 ZIP 根目录，与既有初赛/复赛提交做法一致。
按手册说明，评测系统不对尺寸不匹配的文件做缩放或裁剪，不符合格式的提交直接判无效或计零分，
因此上述尺寸与颜色类型检查是硬门槛。复核不涉及无标注测试集的 mIoU。
