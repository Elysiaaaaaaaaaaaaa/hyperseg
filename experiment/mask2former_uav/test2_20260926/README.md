# H3 复赛测试集 2 推理

以下路径表和日志记录的是 2026-09-26 在数据盘上的原始运行。
系统盘镜像的当前路径与启动方式见 [系统盘迁移记录](../system_image_20260926/README.md)。

**2026-09-26 已完成推理和下载。** 当前本地提交文件为
[swin-l+hyperseg_复赛.zip](swin-l+hyperseg_复赛.zip)（1300 张预测，约 12.73 MiB），
服务器提交检查和本地 ZIP 完整性检查均通过，详见
[结果记录](../experiment.md#2026-09-26h3-复赛测试集-2-推理完成与-zip-回收)。
2026-09-27 依照比赛手册逐张复核的记录见
[manual_compliance_20260927.json](logs/manual_compliance_20260927.json)。

本任务使用 H3（Swin-L + HyperSeg）的验证集最佳检查点，step=137200，
Val mIoU=77.6567936%。复赛测试集只有图像，不能计算 mIoU。

## 路径与设置

| 项目 | server2 路径或设置 |
| --- | --- |
| 输入压缩包 | `/root/autodl-tmp/data/test_2.zip` |
| 输入图像 | `/root/autodl-tmp/data/low_altitude_2026/test_2/images` |
| 检查点 | `/root/autodl-tmp/work_dirs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32/best.pt` |
| 工作目录 | `/root/autodl-tmp/work_dirs/mask2former_uav/h3_test2_20260926` |
| 推理设置 | FP32，512×512 滑窗，50% 重叠，无 TTA |
| 预期输入 | 1300 张 1024×1024 PNG |
| 输出 | 工作目录下 `predictions/` 和 `h3_test2_predictions.zip` |

压缩包 SHA-256：`c8784dba28f19b86e7043bced92e42f98b346ae3f3ea13326f51ef6af295b65a`。

`prepare_data.py` 检查压缩包哈希、ZIP 成员及 CRC，解压后逐张解码图片并验证尺寸，
将文件名清单保存为工作目录的 `data_manifest.json`。不会读取或修改固定训练划分。

## GPU 开启后的启动命令

系统盘迁移后，在 server2 执行：

```bash
cd /root/hyperseg
bash experiment/mask2former_uav/test2_20260926/launch_server2.sh
```

启动器先检查 CUDA，随后使用 nohup 后台运行，并记录 `inference.pid` 和 `runner.log`。
已有预测或已有运行进程时拒绝重复启动。`run_inference.py` 在启动前验证数据清单、
检查点 step/实验名称/通道数，并调用原有 `infer_h3_hyperseg.py` 导出。

仅检查数据、模型元信息和依赖，不执行推理：

```bash
cd /root/hyperseg
PYTHONPATH=/root/mmsegmentation:/root/hyperseg \
  .venv-mask2former/bin/python -u \
  experiment/mask2former_uav/test2_20260926/run_inference.py --check-only
```

## 结果与回收

`runner.log` 每 30 秒记录预测数量，模型输出写入 `infer.log`。
完成后自动执行 `tools/check_submission.py`，要求 1300 个文件与输入同名、
PNG 为单通道 L、1024×1024、像素值 0..8。
通过检查后生成 ZIP（根目录直接存放预测 PNG）以及带 SHA-256 的 `result.json`；
`exit_code=0` 才表示完整成功。

本次已将 `runner.log`、`infer.log`、`submission_check.log`、`result.json`、
`exit_code` 等日志下载到 `logs/`，预测 ZIP 位于本目录。两端 ZIP 哈希一致，
本地复核记录见 `logs/download_verification.json`，运行分析已追加到上级 `experiment.md`。
