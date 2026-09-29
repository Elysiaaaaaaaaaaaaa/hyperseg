# Mask2Former / H3 UAV 实验研究日志

既有 H1、H3 的完整训练记录与结果见 [README.md](README.md)。
本文件记录后续任务及新日志分析。

## 2026-09-26：H3 复赛测试集 2 上传与推理准备

**准备阶段状态：数据及脚本已就绪，当时尚未启动推理。** 用户确认先完成上传和准备，
待 server2 开启 GPU 模式后运行。预检记录时间为北京时间 2026-09-26 11:43。

### 数据校验

- 本地源文件：项目根目录 `test_2.zip`，2140500706 字节。
- server2 压缩包：`/root/autodl-tmp/data/test_2.zip`。
- 两端 SHA-256 一致：`c8784dba28f19b86e7043bced92e42f98b346ae3f3ea13326f51ef6af295b65a`。
- 已解压到 `/root/autodl-tmp/data/low_altitude_2026/test_2/images`。
- 共 1300 张 RGB PNG，全部逐张解码通过，均为 1024×1024；ZIP CRC 检查通过。
- 数据只有图像，没有真值。不能在该数据上报告 mIoU；未修改固定 train/val/test 划分。
- 保留压缩包和解压数据后，数据盘剩余约 2.9 GiB。

### 模型与推理协议

使用 server2 的
`/root/autodl-tmp/work_dirs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32/best.pt`。
实际读取检查点确认：实验标识 `H3_swin_l_hyperseg`，step=137200，9 类输出，
最佳 Val mIoU=77.65679359436035%。历史有标注 Test mIoU=78.72881889343262%，
该历史值不属于本次复赛测试集。

沿用前次无标注导出协议：FP32、512×512 滑窗、50% 重叠、无 TTA。
本地与服务器的 `infer_h3_hyperseg.py` 和 `swin_l_model.py` SHA-256 一致。
新任务调用原有推理代码，不修改模型结构或权重。

### 预检结果与限制

- server2 PyTorch 为 2.3.0+cu121，Swin/MMSeg 模型依赖可导入。
- 准备阶段 `cuda_available=false`，无可用 GPU；CPU 模式内存上限为 2 GiB。
- 首次常规读取约 2.3 GiB 的训练检查点时，预检进程被终止（退出码 137）。
  预检改为 `torch.load(..., mmap=True)` 后正常完成，避免将优化器等张量全部读入内存。
- 数据清单、检查点元信息、依赖和脚本语法检查已通过。没有执行 GPU 前向推理，
  没有生成预测，也没有新的精度结果。

### 已准备的运行与归档

- 脚本与说明：[test2_20260926/README.md](test2_20260926/README.md)。
- 校验日志：[prepare.log](test2_20260926/logs/prepare.log)。
- 文件清单：[data_manifest.json](test2_20260926/logs/data_manifest.json)。
- 模型和环境预检：[preflight.json](test2_20260926/logs/preflight.json)。
- 工作目录：`/root/autodl-tmp/work_dirs/mask2former_uav/h3_test2_20260926`。
- GPU 开启后运行 `bash experiment/mask2former_uav/test2_20260926/launch_server2.sh`，
  后台保存进度和推理日志，完成后自动检查 1300 个预测的同名、单通道、尺寸及标签范围，
  通过后生成 `h3_test2_predictions.zip` 和带哈希的 `result.json`。
- 后续推理完成后回收日志和预测 ZIP，并在本研究日志补充实际运行与提交检查结果。

## 2026-09-26：H3 复赛测试集 2 推理完成与 ZIP 回收

**状态：1300 张预测全部完成，格式检查通过，ZIP 和日志已下载到本地。**
用户启动 GPU 后授权运行；北京时间 11:46:52 通过 GPU 预检，使用 RTX 4090
后台启动任务，PID=1217。采用上述 H3 step=137200 检查点、FP32、512 滑窗、
50% 重叠、无 TTA，未修改推理模型或权重。

### 运行日志分析

- 流程耗时 380.63 秒（约 6 分 21 秒，包含推理、输出检查及打包，不含下载）。
- 进度日志每 30 秒输出一次；模型加载后的各记录间通常新增 113～116 张预测，
  运行过程持续推进。进度行最后记录到 1218，随后完成全部图片并进入校验、打包；
  最终结果以 `result.json` 和提交检查记录的 1300 张为准。
- `exit_code=0`，`result.json` 标记 `status=complete`、`submission_check=passed`。
- `infer.log` 的 Swin 初始化提示来自先构造未加载分类预训练权重的骨干，
  随后原推理脚本严格加载完整 H3 检查点；本次没有进行训练。
- `tools/check_submission.py` 输出 `OK: 1300 predictions`：预测与输入同名，
  全部为 L 模式、1024×1024，标签值均在 0..8。
- 复赛输入没有真值，本次只验证推理完成及提交格式，没有新增 mIoU 或比赛得分。

### ZIP 与本地复核

- 当前本地提交文件：[swin-l+hyperseg_复赛.zip](test2_20260926/swin-l+hyperseg_复赛.zip)。
- server2 结果：`/root/autodl-tmp/work_dirs/mask2former_uav/h3_test2_20260926/h3_test2_predictions.zip`。
- 文件大小：13349726 字节（约 12.73 MiB）；ZIP 根目录直接存放 1300 个预测 PNG。
- SHA-256：`8ebee6e7824f26cd393ee546510b5a3f45e83bd7f90dcd3d8f7b524a160da4be`。
- 下载后重新计算 SHA-256，与服务器一致；ZIP CRC 检查通过，文件名清单与输入完全一致，
  无重复条目；逐文件检查 PNG 头，全部为 1024×1024、8-bit 灰度。

日志与验证依据：

- [runner.log](test2_20260926/logs/runner.log)、[infer.log](test2_20260926/logs/infer.log)
- [submission_check.log](test2_20260926/logs/submission_check.log)
- [result.json](test2_20260926/logs/result.json)、[exit_code](test2_20260926/logs/exit_code)
- [download_verification.json](test2_20260926/logs/download_verification.json)
- [GPU 预检](test2_20260926/logs/preflight.json)；准备阶段的 CPU 预检保留为
  [preflight_cpu.json](test2_20260926/logs/preflight_cpu.json)。

## 2026-09-26：server2 系统盘镜像迁移与校验

**状态：系统盘副本已完成并通过逐文件校验；尚未在网站上保存镜像。**
为便于网站保存系统盘，项目代码及虚拟环境已复制到 `/root/hyperseg`，
MMSegmentation 源码复制到 `/root/mmsegmentation`，比赛数据复制到
`/root/hyperseg/dataset/low_altitude_2026`，H3 最佳权重复制到
`/root/hyperseg/runs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32/best.pt`。
数据盘原件保留，未移动或删除。重复测试 ZIP 和旧 `runs/` 的 LoveDA 等实验输出未复制。

校验覆盖有标注图像 6996 张及同名掩码 6996 张、初赛无标注图像 500 张、
复赛无标注图像 1300 张，四个目录均与源目录逐文件 checksum 一致。
固定划分仍为 train/val/test = 5598/699/699，互斥且覆盖全部有标注图像；
三个划分文件和 H3 最佳权重与源文件哈希一致。权重 SHA-256 为
`83f68e034efee4e8563436f80b90e48d8af2d7260316b1caad82f06be3c62d16`。
系统盘当前约占 20 GiB、剩余 11 GiB。

复制来的虚拟环境原本含有数据盘绝对路径，已修复 activation 和 console scripts；
`pip`、PyTorch 2.3.0+cu121、MMCV 2.1.0、MMEngine 0.10.7、MMDetection 3.3.0、
MMSegmentation 1.2.2 均从系统盘路径可用。`run.py check` 通过，
H3 训练 dry-run 和复赛推理 `--check-only` 显示的数据、代码、权重路径均在系统盘。
预检确认检查点 step=137200、9 类、1300 张复赛图像；当前 server2 为 CPU 模式，
没有在迁移后重新做 GPU 前向或生成新预测。此前完整 GPU 推理结果见上一节。

首次迁移日志有一条空间上界检查的 awk 科学记数法错误，但复制继续完成，
系统盘剩余空间充足；脚本已修正，最终逐文件核对通过。
操作说明及脚本见 [system_image_20260926/README.md](system_image_20260926/README.md)。
依据：[迁移日志](system_image_20260926/logs/migration.log)、
[逐文件校验日志](system_image_20260926/logs/verification.log)、
[校验结果](system_image_20260926/logs/verification.json)、
[系统盘复赛预检](system_image_20260926/logs/preflight.json)。

## 2026-09-27：复赛预测 ZIP 对照比赛手册复核

依据 [赛题手册第十三节](../../doc/无人机低空航拍图像语义分割_纯文本.md)，
复赛测试集 2 的输出格式与初赛相同：预测图须与输入同名、为单通道灰度 PNG、
尺寸 1024×1024、像素值属于 0..8，并压缩成一个 ZIP。

对当前文件 `test2_20260926/swin-l+hyperseg_复赛.zip` 重新检查：
SHA-256 为 `8ebee6e7824f26cd393ee546510b5a3f45e83bd7f90dcd3d8f7b524a160da4be`，
与运行结束时校验过的原始结果一致。ZIP 无损坏，内含 1300 个根目录 PNG、
无重复文件名；与官方测试集 2 的 1300 个文件名逐一相同。
用 Pillow 逐张完整解码，全部为 `L` 灰度模式、1024×1024，
每张的像素范围均在 0..8；不存在 RGB、Palette 或超范围标签。
因此当前 ZIP 的**提交文件格式符合手册要求**。本次核验不涉及无标注测试集的 mIoU。

核验明细：[manual_compliance_20260927.json](test2_20260926/logs/manual_compliance_20260927.json)。
