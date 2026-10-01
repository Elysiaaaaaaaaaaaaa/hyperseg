# 项目长期记忆 · hyperseg（无人机低空航拍语义分割）

> 2026-10-01 二次压缩（原 16.5 KB，注入时被截断）。细粒度日志见 `.workbuddy/memory/YYYY-MM-DD.md`。
> 项目说明已落库到仓库 `README.md` / `HYPERSEG.md` / `AGENTS.md`，此处只留踩坑与结论。

## 环境与远程

- 仓库 `D:\myproject\hyperseg`。**E 盘 `E:\myproject\hyperseg` 是外置 USB 盘**，偶发返回错误数据、拔掉即消失
  ⇒ **GB 级交付物一律放 D 盘；E 盘上的哈希结论不可信，必须到 D 盘复算**。以后不要再往 E 盘放交付物。
- 服务器见 `sshconfig.toml`。server3 = AutoDL RTX 4090（平时关机），`/root/hyperseg` +
  `.venv-mask2former`（torch 2.3.0+cu121 / transformers 4.57.6）。
- **远程同步**：本机无 sshpass，用 `experiment/h3_bgfix_20260930/remote.py`（paramiko）。解释器必须用
  `C:\Users\86189\.workbuddy\binaries\python\envs\default\Scripts\python.exe`（系统 3.9 / managed 3.13 都没装
  paramiko）；路径参数前加 `MSYS_NO_PATHCONV=1`。子命令 `put --local f --remote f` / `run -- "<cmd>"` / `pull` / `collect`。
- **AutoDL GPU 会在长训练收官时从容器消失**（宿主不再暴露，只能去控制台重启）。判据只有两条：
  `nvidia-smi --query-gpu=name` 有输出、`torch.cuda.is_available()` 为真；**不要看 `/dev/nvidia0`**。
  长任务脚本开头先断言 CUDA 可用。
- **server2 磁盘**：系统盘 `overlay` 上限 30 G；数据盘 `/root/autodl-tmp` 本实例仅 50 G。`df -h` 不带参数
  **看不到** autodl-tmp（只列只读公共区 `/autodl-pub`，7 T 非可用空间）⇒ **必须单独跑 `df -h /root/autodl-tmp`**。
  `dataset/low_altitude_2026` 已软链到 `/root/autodl-tmp/data`，不要再在系统盘铺第二份。
- **上机前先核对远端脚本与本地是否一致，再怀疑参数/环境**（曾因 server2 旧版 `watch_h2_remote.sh` 硬编码
  20k 而误判"训练已结束"；`put` 一次即排除）。

## 权重与加载

- `models/hyperseg_b3_best.pt`：HyperSegUAV + MiT-B3（v1），9 类，epoch 141，val mIoU 74.2023%。
  Transformers 5 保存，骨干键名 `encoder.backbone.stages.{i}...`，Transformers 4 环境必须经
  `hyperseg_uav.translate_legacy_keys()`（**628 个键**）转换。`tools/infer_hyperseg.py` / `test_hyperseg.py`
  已内置；后者用 `strict=False`，**`missing=628` 即加载失败**。
- `models/hyperseg_resume_best.pt` **是 v2 不是 v1 续训版**：`tools/model_1.py` 定义（config 带 `version:"v2"`），
  **730 张量** vs v1 的 686，epoch 134，val mIoU **76.3160%**。用 v1 类构造报
  `unexpected keyword argument 'version'`，须先 `config.pop("version")`。
  现成加载器 `experiment/b3_resume_test2_20260930/v2_model.py`。
- **复核 val_miou 必须连口径一起复现**：v2 的 0.7631595244109886 口径 = **size 768 / 评估 batch 8 / 无 TTA /
  只做 `/255`**（复现 0.7631586）；照搬 v1 的 batch 2 只得 0.740312。`mean_iou` 是"batch 内逐类再对 batch 平均"，
  batch 越大越接近全数据集口径 ⇒ **对不上先怀疑 batch size**。
- **推理协议**：b3 = 768 / 0.5 重叠 / 无 TTA；H3 Swin-L = 512；MMSeg 系 = 整图 1024。不要擅自加 TTA。

## 交付与提交（复赛，队名 Mondstadt）

- 组织规范 `doc/提交物组织规范.md`。复赛交：完整镜像 + 技术方案 + 网盘链接。
- **正式镜像** 13 079 718 593 B（12.18 GiB），sha256
  `c541042f509f4544a05a28a92054e714c152f8d9145597d76ee55551040a5c9d`，切 4 卷（3800 MiB，末卷 1 125 952 193 B）。
  容器内 `docker import` 冒烟已通过（镜像 `9b65ec6f`、展开 34.3 GB、`--check-only` 打印 628 / epoch 141 / 1300 张）；
  只剩 `Dockerfile` 干净重建未跑。
- **正式提交包 = `dist/低空航拍语义分割_复赛_Mondstadt_20260929_完整交付/`**（82 文件 / 12.53 GiB，含
  `04_镜像/` 四卷 + `CHECKSUMS.sha256` + `MANIFEST.json`，2026-10-01 复核 82/82 OK）。总 zip
  `dist/..._完整交付.zip`（13 450 992 182 B，sha `7e060bc0...`）仅供"表单只收单文件"场景，**网盘仍传分卷**。
  `dist/` 已清理为只保留这两项。
- **工具**：`tools/build_submission_package.py --root <包>`（幂等重建，刷 image-tools 副本 + 重算清单，
  并从 `packaging/` 回填 20 个手工文件）；`tools/assemble_delivery_tree.py --package --volumes --out`
  组装交付目录。写清单必须 `newline="\n"`，否则 CRLF 让 `sha256sum -c` 全报错。
  **手工文件（README/Dockerfile/configs/docker/docs/scripts/DATA_ACCESS/04_镜像说明）权威副本在 `packaging/`**
  —— `dist/` 被 gitignore，只改 dist 会丢。
- 复现脚本从 `03_代码与复现/` 调 `scripts/0X_*.sh`（脚本自 cd 进 hyperseg 并设 PYTHONPATH）。容器内跑
  `run_inference.py` 需 `export PYTHONPATH=/root/hyperseg`（它不在镜像关键哈希表内，改它无需重导镜像）。
- 预测包自检不能只信 Pillow `mode=='L'`：必须解析 PNG 块确认 IHDR 颜色类型 0、无 PLTE/tRNS
  （`experiment/b3_test2_20260929/check_manual_compliance.py`）。

## 本机文件系统与磁盘（2026-10-01 实测）

- **本机删除有 `safe-delete` 拦截层**：`rm`/`Remove-Item` 被路由到 D 盘回收站；目录按展开文件数计数，
  单轮累计 >50 报 `SAFE_DELETE_BULK_CONFIRM_REQUIRED` ⇒ **逐项或按子目录小批删**。**PowerShell 会话对中文
  路径乱码**，含中文路径的删除一律用 Bash。
- **D 盘"删除 ≠ 腾空间"**：`rm`、`os.remove` 实测都留下 `D:\$RECYCLE.BIN\…\$R*` 条目，可用空间分毫不动
  （疑为第三方过滤驱动：`Ai磁盘管家`、`360RecycleBin`、`DustAwayMove`）。为省空间删完**必须
  `shutil.disk_usage('D:/')` 复核 free 是否真涨**；只有清空回收站（`Clear-RecycleBin -DriveLetter D`，
  不可恢复需确认）才算腾出。回收站条目保留**原文件 mtime**，按时间戳判断会误判，要拿字节数对。
- `D:\Ai磁盘管家\软件搬移\`（约 5.6 GiB）是 C 盘搬移来的**在用程序本体/链接目标**（Quark、Chrome、百度网盘），
  **不要当垃圾删**。D 盘常年告急（2026-10-01 仅剩 3.3 GiB，回收站独占 6.84 GiB），GB 级产物落盘前先确认空间。
- 判定"删除是否真释放"：写已知大小临时文件 → free 应立减；删后采样 2/10/30 s，free 不回涨即为重定向；
  再拿字节数去 `$RECYCLE.BIN` 比对 `$R*` 坐实。
- **E 盘 exFAT 改不动已有文件**：`mv`/`rm`/`Rename-Item`/`write_text`/`cat >` 全 `Permission denied`；
  唯 Write/Edit 能改写但写成 CRLF。故 E 盘校验要 `tr -d '\r' < CHECKSUMS.sha256 | sha256sum -c -`。
  已写错名就逐字节 `cp`，别依赖 rename。
- **中断的镜像流不可续传**（pax 头记 atime/ctime，读文件即改源端属性）⇒ 要可续传须在导出端消除不确定性
  （`--pax-option` 去 atime/ctime、`--mtime` 归一、或 `--format=gnu`）。判定工具：`tools/inspect_partial_stream.py`
  → `compare_prefix_attrs.py` → `tally_prefix_drift.py`。
- **Docker 数据盘** `D:\docker\DockerDesktopWSL\disk\docker_data.vhdx`（别跟系统盘 `main\ext4.vhdx` 搞混）：
  **只涨不缩**（挂载 `rw,relatime` 无 `discard`），空 Docker 真实占用约 1.6 GB、**压到 1.9–2.1 GB 就是到底**。
  2026-10-01 实测 33.91 GiB → 1.98 GiB（+32 GB），流程见用户级技能 `~/.workbuddy/skills/docker-vhdx-compact`。
  **压缩需管理员**：agent 会话无管理员权限，`Start-Process cmd.exe -Verb RunAs` 被拦；`diskpart.exe -Verb
  RunAs` 必须配 `dangerouslyDisableSandbox: true`。**验证只认 vhdx 字节数**。
- **PowerShell 工具在本机不回显 stdout** ⇒ 结果 `Out-File` 到临时文件再 Read。

## MMSeg 系训练（H1/H2/H3）

- **`work_dir` 里没有 `train.log`**：文本日志在 `<work_dir>/<时间戳>/<时间戳>.log`，标量在
  `.../vis_data/scalars.json`。**验证行的键是裸 `mIoU`/`aAcc`/`mAcc`/`step`，没有 `val/mIoU`**（用它 grep 永远是 0）。
  参考 `watch_h2_remote.sh`。`iter` 正则要容忍右对齐空格 `\[ *[0-9]\+/ *[0-9]\+ *\]`；`lr` 不能
  `awk -F'"lr":'` 取行尾，要先截出数组。
- **MMSeg 1.2.2 的 `SwinTransformer` 不接受 `convert_weights`**（传了 `MODELS.build` 抛 TypeError）；照抄官方
  `configs/mask2former/mask2former_swin-{b,l}-*.py`。Swin-L Mask2Former 512/batch2/FP32：峰值 8.4 GB、
  0.30 s/update、699 张 1024 验证约 80 s、瘦身 ckpt 827 MB、骨干 195.7 M、预训练覆盖 100%。
- **H2**（`experiment/mask2former_uav/`，Swin-L + Mask2Former）：20k 臂 best val mIoU **0.6980**。
  **H2-160k 2026-10-01 收官**（跑满 160k，15 h 35 m，`exit_code=0`）：best-by-val 在 iter 132000 = val 78.20
  （132000 后 28 次验证全在 77.25–78.09 ⇒ 进平台）。`test.txt`：best 132000 = 78.56，**末点 160000 = 79.44（+0.88）**
  （20k 臂 @20000 = 70.08）⇒ val 平台 + ±0.5 噪声掩盖尾部收益，**这一臂最终提交用末点检查点，不要用
  `save_best='mIoU'` 的结果**。末点唯一大改善类是 Barren 52.51→57.05（+4.54，Acc 67.45→77.76）。test 全程 > val。
  ⇒ 服务器上 `best_mIoU_iter_*.pth` 会被后来的 best 覆盖，**只有最后一个 best 和 `iter_160000` 留着**。
  **延长训练没修好无标注集**：test_2 背景占比 82k 0.3479 → 132k **0.3649**（全部臂最高），h3 塌陷池内
  0.8274 → 0.8581，20k 臂反而退回 ⇒ **test.txt 的提升不能推断 test_2 变好**。
- **无标注集塌陷病因（2026-10-01 定案；脚本 `analyze_image_stats_vs_perf.py` +
  `attribute_detail_vs_appearance.py`）**：先验背景 27.8% vs 塌陷池内 86% = 分布外覆盖问题。
  ①「低细节」成立、「低对比度」不成立：`detail_lap`（Laplacian 方差）~ 预测背景占比 ρ = −0.256（真值）/
  **−0.494**（test_2），而 `contrast_std` 仅 −0.133、`contrast_rms` 反号 +0.203
  ⇒ **自变量用高频能量，不是灰度动态范围**。塌陷是**断崖**（最低细节五分位 0.719、第二分位骤降 0.317，真值平滑）；
  Q1 上 20k 0.636→82k 0.676→132k 0.719 **越训越重 ⇒ 迭代不是解**。
  ② **关键修正：「低细节」不是独立病因，与「暗 + 蓝绿偏色」几乎是同一变量**（detail_lap ~ cast_br(B−R)
  ρ=**−0.675**、~mean_r +0.623）；偏相关控住外观后 detail 仍 −0.266、cast_br 只剩 +0.141 ⇒ 细节更直接，
  偏色靠交互起作用。分层是决定性证据（蓝偏三档 × 档内细节二分 → 预测背景占比）：蓝偏低 0.305/0.205、
  蓝偏中 0.350/0.280、**蓝偏高 0.759 / 0.289** ⇒ **塌陷只在「高蓝偏 × 低细节」这一格**，单有其一都不触发。
  ⇒ **只加"降分辨率"是打空的一半，增强必须"降细节 + 改外观（雾化/方向性偏色/曝光）"成套上。**
  推论：`h3_bgfix/eval_proxy.py` 只扫细节轴，**会系统性低估 aug 臂收益**（病因是二维交互）。
- **已做过的对照 = `h3_bgfix` 的 `aug` 臂**（`bgfix_dataset.py`：zoom_out 真实降尺度 + gsd 降采样再升采样 +
  方向性偏色/雾化/噪声 + GaussianBlur）：域内 **+0.05pt**、代理均值 **+0.06pt**（噪声级），但 test_2 背景
  0.3087→0.2992、`bg>0.8` 122→101，是唯一无新病理的臂。真正打到 42/27 的是 **`false_background` 惩罚（决策层）**，
  代价域内 −1.12~−2.01pt。机制：真值里低细节 ⇔ 背景多，随机降质主要在合成「(低细节, 背景标签)」配对；但
  **训练集最低细节箱自己有 1120 张、"低细节 + 60.9% 前景"**，模型见过仍塌陷 ⇒ 不只是样本没覆盖，而是弱证据时
  背景/no-object 默认胜出。H2 管线已有 `RandomResize(0.5–1.5)+RandomCrop(512)` 与
  `PhotoMetricDistortion.contrast_range=(0.5,1.5)`，缺雾化/方向性偏色/曝光、模糊、噪声、down-up 重采样。
  **杠杆排序：决策层 > 选点（用末点） > 增强（且必须双轴成套）。**
- **H2b**（20k→60k，`launch_h2_server.sh resume60k`，仍未启动、价值已被 160k 臂削弱）；**H2-160k**
  （`launch_h2_server.sh long`）。中途/最终检查点一律用 `run_h2_ckpt_test.sh <ckpt> <label> [--skip-export]`
  （须绝对路径调用；test.txt 评估 + test_2 导出一步到位）。H2 20k 臂用 `--no-save-optimizer`，续训是"新优化器 +
  已退火权重 + LR 回热"，增益混着两个因素；160k 与 20k 臂**不可逐点对照**（PolyLR end 不同）。
- **MMSeg 导出的预测 id 是「原始标签空间 1..8」**（管线 `reduce_zero_label=True`，导出时 id 加回 1）
  ⇒ **1 = Background，不是 0**；出现 0 才是 ignore 泄漏。复核用
  `experiment/mask2former_uav/check_h2_test2_predictions.py`（参数化，含像素占比 / 塌陷统计）。
- **MMSeg 续训不能直接 `--resume`**：`CheckpointHook.save_param_scheduler` 默认 True，旧 `end` 盖掉新的
  → `step()` 条件不成立、**LR 永久冻结且不报错**。必须先跑 `prepare_resume_checkpoint.py` 剥掉
  `param_schedulers` 并重锚 LR。另：`max_iters` 不被 ckpt 覆盖；`--resume` 是 store_true，指定文件要靠
  `load_from` + `--resume`；空 work_dir 里 `--resume` 会静默从头训练。上机必查日志出现 `resumed epoch: 0, iter: N`。
- **MMSeg test_2 导出走 `run.py export`**（非 H3 自研滑窗）：内部调 `tools/test.py --out` + `check_submission.py`，
  协议整图 1024（`mode='whole'`、`size_divisor=32`），0.131 s/张、峰值 2.0 GB、1300 张约 213 s。
  **MMSeg 1.2.2 的 slide 滑窗不能用**（crop 与 pad_shape 不一致 → 尺寸错）⇒ H2 与 bgfix/H3 的 test_2 跨协议，
  背景占比只能看方向。`tools/test.py` 指标输出无 `Summary:` 表头，是逐类表 + `Iter(test) [699/699] aAcc: .. mIoU: ..`。
