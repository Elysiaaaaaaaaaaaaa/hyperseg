# HyperSeg-B3 复赛测试集 2 推理

本任务在 **server3**（AutoDL，RTX 4090 24 GiB）上，用仓库根目录的
`models/hyperseg_b3_best.pt` 对复赛无标注测试集 `low_altitude_2026/test_2/images`
（1300 张 1024×1024 PNG）做滑窗推理，并把预测打包下载回仓库。

**2026-09-29 已完成。** 本地提交文件为
[hyperseg_b3_test2_predictions.zip](hyperseg_b3_test2_predictions.zip)
（1300 张预测，12 655 462 字节 ≈ 12.07 MiB），服务器提交检查和本地 ZIP 复核均通过。
运行与分析见 [experiment.md](experiment.md)。

## 检查点

| 项 | 值 |
| --- | --- |
| 文件 | `models/hyperseg_b3_best.pt` |
| SHA-256 | `fb124302bafaec18aaa3f695dc021efead7bd00f710348d040888bb270148968` |
| 结构 | `HyperSegUAV`，MiT-B3 骨干，9 类，`widths=[64,128,320,512]`，`context_dim=128` |
| 训练轮次 | epoch 141 |
| 记录 Val mIoU | 74.2023%（768×768，固定 `runs/splits/val.txt`） |

## 路径与设置

| 项目 | server3 路径或设置 |
| --- | --- |
| 输入图像 | `/root/hyperseg/dataset/low_altitude_2026/test_2/images` |
| 检查点 | `/root/hyperseg/models/hyperseg_b3_best.pt` |
| 工作目录 | `/root/hyperseg/runs/b3_test2_20260929` |
| 推理设置 | FP32，768×768 滑窗，50% 重叠，无 TTA |
| 每张图的窗口数 | 4（stride 384，覆盖 1024×1024） |
| 输出 | 工作目录下 `predictions/` 与 `hyperseg_b3_test2_predictions.zip` |

768×768 与 `tools/train_hyperseg.py` 的训练、验证分辨率一致，因此导出掩码与
checkpoint 被选中时的尺度相同。复赛测试集没有真值，**不能报告 mIoU**。

## 复赛预测与旧权重键名

`models/hyperseg_b3_best.pt` 由 Transformers 5 保存，骨干编码器键名形如
`encoder.backbone.stages.{i}.blocks.{j}...`；本项目环境的 Transformers 4
使用 `encoder.backbone.encoder.block.{i}.{j}...`。仓库中
`experiment/loveda_fewshot/train.py` 与 `experiment/fewShot_SUIM/run.py` 早已存在
`legacy_encoder_key` 转换，本次把它收拢到 `hyperseg_uav/model.py`
（`legacy_encoder_key` / `translate_legacy_keys`），并让 `tools/infer_hyperseg.py`、
`tools/test_hyperseg.py` 在加载前先做转换；否则这两个工具会用
`missing=628` 静默跳过整个骨干，导出无意义结果。

## 运行命令

```bash
ssh -p 31067 root@connect.bjb1.seetacloud.com      # server3
cd /root/hyperseg
.venv-mask2former/bin/python -u experiment/b3_test2_20260929/run_inference.py --check-only
bash experiment/b3_test2_20260929/launch_server3.sh
```

启动器先检查 CUDA，再用 nohup 后台运行并记录 `inference.pid`、`runner.log`；
已有预测或已有运行进程时拒绝重复启动。`run_inference.py` 在推理前校验输入清单
（1300 张、1024×1024、无重名）、检查点标识与严格加载，完成后自动执行
`tools/check_submission.py`（同名、单通道 L、1024×1024、像素 0..8），
再打包 ZIP 并写入带 SHA-256 的 `result.json`；`exit_code=0` 表示完整成功。

推理前的权重对照复核（有标注验证集上复现训练时的 74.2023%）：

```bash
cd /root/hyperseg
.venv-mask2former/bin/python -u experiment/b3_test2_20260929/val_sanity.py
```

本地复核下载到的 ZIP：

```bash
python experiment/b3_test2_20260929/verify_download.py
```

按比赛手册第十三节的格式复核（直接解析 PNG 的 IHDR 颜色类型，排查调色板与透明块）：

```bash
python experiment/b3_test2_20260929/check_manual_compliance.py
```

判定为 PASS，明细见 `logs/manual_compliance_20260929.json`。

## 结果与回收

- 服务器 ZIP SHA-256：`ff0a2fb0e0580bafb1b7f41f82d29861ad42bf181689f7efff43df3e84b0ef8a`，
  与下载后本地重算一致。
- `logs/` 收录 `runner.log`、`infer.log`、`submission_check.log`、`result.json`、
  `exit_code`、`preflight.json`、`val_sanity.json`、`download_verification.json`、
  `manual_compliance_20260929.json`。
- 需要重跑时先删除服务器工作目录下的 `predictions/`，否则启动器会拒绝启动。
