# 项目长期记忆 · hyperseg（无人机低空航拍语义分割）

## 目录与机器

- 本地仓库 `D:\myproject\hyperseg`；E 盘 `E:\myproject\hyperseg` 存放**大体积交付物**
  （系统盘镜像导出、汇总包），不要把 GB 级文件放 D 盘仓库。
- 服务器：server1/2/3 见 `sshconfig.toml`。server3（AutoDL，RTX 4090）已含
  `/root/hyperseg` 全套代码、`.venv-mask2former`（torch 2.3.0+cu121 / transformers 4.57.6）、
  `dataset/low_altitude_2026`；**平时关机**，用前需在控制台开机。
  本地代码同步到服务器靠 `sshpass + scp`，远程 python 一律 `ssh 'python -' < script.py`。

## 权重与加载（易踩坑）

- `models/hyperseg_b3_best.pt`：HyperSegUAV + MiT-B3，9 类，epoch 141，记录 val mIoU 74.2023%。
- 该检查点由 **Transformers 5** 保存，骨干键名为 `encoder.backbone.stages.{i}.blocks.{j}...`；
  当前环境是 Transformers 4，必须经 `hyperseg_uav.translate_legacy_keys()`（628 个键）转换后才能加载。
  `tools/infer_hyperseg.py`、`tools/test_hyperseg.py` 已内置转换。
  `test_hyperseg.py` 用 `strict=False`，缺键不报错，看到 `missing=628` 一律视为加载失败。
- 推理协议：b3 模型 768×768 / 0.5 重叠 / 无 TTA（与训练分辨率一致）；H3 Swin-L 用 512。不要擅自加 TTA。

## 交付与提交

- 当前阶段是**复赛**（预测测试集 2，1300 张）；提交物结构、分卷规则、网盘上限、
  现有镜像资产核对结论见 `doc/提交物组织规范.md`（原文件名含"半决赛"，已改名）。
  复赛额外要交：完整镜像文件 + 技术方案 + 百度网盘链接。
- 现有可用镜像底座：`E:\myproject\hyperseg\server2-system-no-dataset-20260926.rootfs.tar.gz`
  （12.83 GiB，sha256 `e8e4ffae…3cee8e`，已排除数据集，含 b3 权重与固定划分），
  但代码是 09-26 快照、**不含 09-29 的旧键名修复**，不能直接用于复现提交，需在 server3 上补齐后重导。
- 预测包自检：不能只信 Pillow 的 `mode=='L'`，手册严禁调色板索引色，
  必须解析 PNG 块确认 IHDR 颜色类型为 0、无 PLTE/tRNS。见
  `experiment/b3_test2_20260929/check_manual_compliance.py`。
- 队名已定 **Mondstadt**（2026-09-30 起所有产物统一用它）。
- 技术方案本轮以 **docx** 交付：源稿 `dist/技术方案_源.html`，产物为
  `02_技术方案/Mondstadt_技术方案_无人机低空航拍图像语义分割.docx`，
  **权威副本在仓库 `packaging/02_技术方案/`**。
  生成链路走官方 `html-to-docx`，**不要用 Edge 无头导 PDF**（该文档会让它静默失败）。
  命令与 Windows venv 绕过见 skill `hyperseg-server-inference` 第 10 条。
- 镜像流式下载见 `tools/download_rootfs.py`（账本在 `runs/<label>/`）。
  **E 盘是外置 USB 盘：拔掉就整盘消失，而且会偶发返回错误数据**（2026-09-30 实测：
  同一条 `cat part_* | sha256sum` 两次得到不同哈希，D 盘连测 3 次稳定；
  `cmp` 判「逐字节相同」却给出不同 sha256）。**GB 级交付物一律放 D 盘内盘，
  E 盘上得到的哈希结论不可信、必须到 D 盘复算。**
  导出脚本与流助手一律放**排除区** `/root/autodl-tmp/`，既不进镜像也不扰动 tar 头。
- **中断的镜像流不可续传**（2026-09-30 实测判定，勿再尝试手工修复）：导出用
  `tar --format=pax`，pax 头把 **atime/ctime 也按纳秒记录**；tar 读文件本身在 relatime 下就会
  把 atime 顶成当时时刻，且 **ctime 任何用户态工具都设不回去**。一次导出就永久改变了源端属性，
  第二次导出必然对不上。要可续传必须在导出端消除不确定性（`--pax-option` 去掉 atime/ctime、
  `--mtime` 归一化、或退回 `--format=gnu`）。判定工具：`tools/inspect_partial_stream.py` →
  `tools/compare_prefix_attrs.py` → `tools/tally_prefix_drift.py`。
- 复赛镜像正式产物：**13 079 718 593 字节（12.18 GiB），sha256
  `c541042f509f4544a05a28a92054e714c152f8d9145597d76ee55551040a5c9d`**；
  已切 4 卷（3800 MiB，末卷 1 125 952 193 B）。**交付副本在 D 盘**：
  `dist/低空航拍语义分割_复赛_Mondstadt_20260929_完整交付/04_镜像/`。
  `cat part_* | sha256sum` 复算一致（无需写出合并副本即可验证）。
  容器内 `docker import` 冒烟**已通过**（2026-09-30：镜像 `9b65ec6f9121`、展开 34.3 GB、
  `--check-only` 打印 `628` / epoch 141 / val_miou 0.7420226665631505 / 1300 张；
  转写 `experiment/b3_test2_20260929/logs/container_smoke_20260930.log`）。
  仍未跑的只有 `Dockerfile` 干净重建。
- **容器内跑 `run_inference.py` 需要 `export PYTHONPATH=/root/hyperseg`**：
  `python experiment/<x>/run_inference.py` 只把脚本自身目录放进 `sys.path`。
  仓库/提交包里的副本已加 `sys.path` 自引导（与 `val_sanity.py`、`tools/infer_hyperseg.py` 一致），
  **镜像里那份是加之前的**；`run_inference.py` 不在镜像关键文件哈希表内，改它无需重导镜像。
- E 盘 exFAT 上**改不动已有文件的任何方式**：`mv` / `Rename-Item` / `rm` /
  `write_text` / `cat > f` 全是 `Permission denied`（沙箱内外一样）；
  只有 Write/Edit 工具能改写，而它们**会把换行统一成 CRLF**。
  因此 E 盘上的 `CHECKSUMS.sha256` 是 CRLF，`sha256sum -c` 会因文件名尾部 `\r` 失败 ——
  校验命令一律写 `tr -d '\r' < CHECKSUMS.sha256 | sha256sum -c -`。
  落盘不要依赖 rename，已写错名就用**逐字节复制**（`cp` 可用）。
- 提交包由 `tools/build_submission_package.py --root <包目录>` 重建（幂等，会刷新
  `03_代码与复现/image-tools/` 的脚本副本并重算 `CHECKSUMS.sha256` + `MANIFEST.json`；
  还会从仓库 `packaging/` 镜像回 20 个手工文件）。
  **手工文件（README、Dockerfile、configs、docker、docs、scripts、DATA_ACCESS、
  04_镜像 说明）的权威副本在仓库 `packaging/`** —— 因为 `dist/` 被 gitignore，
  只改 dist 的话重建或换机就丢了。
  复现脚本约定：从 `03_代码与复现/` 调用 `scripts/0X_*.sh`，脚本自己 `cd` 进 `hyperseg/`
  并设好 `PYTHONPATH`（原先 `cd ..` 少了一级，五条脚本都跑不通，已修）。
  轻量包**不含镜像分卷**，`04_镜像/` 只放说明与清单。
  2026-09-30 状态：**78 文件 / 0.35 GiB，`sha256sum -c` 78/78 OK**
  （含新补回的 `models/nvidia--mit-b3/` —— 仓库原先缺 `config.json` 与
  `model.safetensors`，`hyperseg_uav/model.py` 会因此回落 Hub，需联网）。
- **完整交付目录**由 `tools/assemble_delivery_tree.py --package <包> --volumes <分卷目录>
  --out <交付目录>` 组装：包 + 分卷 + 全树 `CHECKSUMS.sha256`/`MANIFEST.json` +
  `04_镜像/CHECKSUMS.sha256`（两个清单都由脚本**生成**，不复制现成文件）。
  产物：`dist/低空航拍语义分割_复赛_Mondstadt_20260929_完整交付/`，**82 文件 / 12.53 GiB**。
  写清单必须 `write_text(..., newline="\n")`，否则 Windows 写成 CRLF，
  `sha256sum -c` 全报 `No such file or directory`。
- 另有一份**总压缩包**（只为"提交表单只收单文件"场景）：
  `dist/低空航拍语义分割_复赛_Mondstadt_20260929_完整交付.zip`，13 450 992 182 B，
  sha256 `7e060bc08991f421c55565a581d79e48752c44fd050de5949ace19434145855b`。
  用 7-Zip `a -tzip -mx0`（STORED，内层已是 gzip 分卷，再压零收益）。
  **网盘上传仍传分卷**，不要把 12.53 GiB 的 zip 当上传物。

