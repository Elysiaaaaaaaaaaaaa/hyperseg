# Docker 环境验证记录

## 2026-09-28：Windows Docker Desktop 启动检查

按用户要求仅检查容器启动与依赖，不运行推理或训练，不加载模型权重。

- 镜像：`server2-system:no-dataset-20260926`。
- Docker Desktop Linux 引擎启动后，默认 CMD 可以启动容器并执行 shell 命令；工作目录为 `/root/hyperseg`。
- 默认 Python 为 `/root/hyperseg/.venv-mask2former/bin/python`，版本 3.12.3。
- torch 2.3.0+cu121、numpy 2.5.3、Pillow 10.3.0、mmcv 2.1.0（含 mmcv.ops）、mmengine 0.10.7、mmseg 1.2.2、transformers 4.57.6 和项目模型模块均成功导入。
- 推理入口文件与 best.pt 存在；权重文件大小 2369198126 字节。本次未检查权重反序列化或加载兼容性。
- 使用 `--network none` 验证，不依赖在线下载；临时检查容器使用 `--rm` 自动清理。
- 检查脚本首次执行遇到 PowerShell 管道 CRLF 使 shell 将 exit 识别为带回车的命令；已修正测试脚本并重新执行，此问题来自检查脚本，不是镜像。

结论：容器启动与关键依赖导入检查通过。尚未验证 GPU、模型推理、预测格式或比赛平台执行兼容性；官方附件下载 404 的原因不在本次检查范围内。

复现脚本：`startup_check.ps1`、`startup_check.py`；完整输出：`startup_check.log`。
