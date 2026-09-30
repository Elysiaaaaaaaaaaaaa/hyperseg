# 无人机低空航拍图像语义分割 · 复赛提交包

队伍名：`Mondstadt`　阶段：**复赛**（测试集 2，1300 张）　打包日期：2026-09-29

本包按 `doc/提交物组织规范.md` 组织，是**完整交付形态**：代码、技术方案、预测结果与
**镜像分卷本体**同处一棵目录树，整体上传网盘即可，评审侧无需再从别处取镜像。
镜像的来源、合并方式与容器内验收步骤见 `04_镜像/README_IMAGE.md`。

> 若平台另有轻量代码包入口，可把 `04_镜像/` 内的 4 个分卷移除后单独上传，
> 其余文件不受影响；`README.md` 与 `04_镜像/{README_IMAGE.md,image_manifest.json}`
> 已写明镜像的获取与校验方式，不依赖分卷是否与本包同处一地。

---

## 一、要求对照表

| 组委会要求（复赛） | 本包对应产物 |
| --- | --- |
| 在当阶段测试集上预测，输出格式同初赛，压缩为一个 zip | `01_预测结果/predictions_test2_round2.zip`（1300 张，与输入同名） |
| — 格式自检 | `01_预测结果/submission_check.json`、`file_list.txt` |
| 完整镜像文件 | `04_镜像/hyperseg-rootfs-20260930.tar.gz.part_01` … `part_04`（4 卷，合计 13 079 718 593 B / 12.18 GiB）+ `README_IMAGE.md` + `image_manifest.json` |
| 技术方案 | `02_技术方案/Mondstadt_技术方案_无人机低空航拍图像语义分割.docx` |
| 可复现的 docker 文件（完整训练 + 验证代码、复现脚本） | `03_代码与复现/`（`Dockerfile` + `hyperseg/` + `scripts/` + `configs/` + `docker/`） |
| 代码、环境、操作说明文档 | `03_代码与复现/docs/{ENVIRONMENT,OPS,REPRODUCE}.md` |
| 数据来源与获取方式 | `06_数据说明/DATA_ACCESS.md` + `splits/` |

## 二、目录

```text
低空航拍语义分割_复赛_Mondstadt_20260929/
├── README.md                  本文件
├── CHECKSUMS.sha256           包内所有文件的 sha256（含镜像分卷；sha256sum -c 可校验）
├── MANIFEST.json              机器可读清单：路径 / 字节数 / sha256
├── 01_预测结果/                测试集 2 的预测 zip + 文件清单 + 格式自检
├── 02_技术方案/               技术方案（算法动机、方法、伪代码）
├── 03_代码与复现/              源码 + Dockerfile + 复现脚本 + 环境与操作文档
├── 04_镜像/                   镜像本体（4 个分卷）+ 交付说明 + 清单 + 分卷校验清单
├── 05_实验证据/                推理/校验日志与指标 json
└── 06_数据说明/                数据获取方式 + 固定划分的文件名清单
```

## 三、关键数字

| 项目 | 值 |
| --- | --- |
| 模型 | HyperSeg-UAV，SegFormer **MiT-B3** 骨干 + 自定义 decoder/fusion/head |
| 检查点 | `models/hyperseg_b3_best.pt`，epoch 141，SHA-256 `fb124302bafaec18aaa3f695dc021efead7bd00f710348d040888bb270148968` |
| 记录 val mIoU | 0.7420226665631505（74.2023 %，固定有标注验证划分） |
| 复现 val mIoU | 0.7420232149377345（差 5.5e-7） |
| 推理协议 | FP32，768×768 滑窗，50 % 重叠，**无 TTA**，单模型（无 ensemble） |
| 预测 zip | 1300 张，12 655 462 字节，SHA-256 `ff0a2fb0e0580bafb1b7f41f82d29861ad42bf181689f7efff43df3e84b0ef8a` |
| 镜像（合并后） | 13 079 718 593 字节（12.18 GiB），SHA-256 `c541042f509f4544a05a28a92054e714c152f8d9145597d76ee55551040a5c9d` |
| 推理耗时 | 224.83 s / 1300 张（RTX 4090） |

**无标注测试集上不报告 mIoU**：测试集 2 没有真值，本包给出的 mIoU 一律来自固定的
有标注验证划分。`05_实验证据/metrics/summary.json` 里写明了这一点。

## 四、一键复现（推理路径）

```bash
# 环境：二选一
#   a) 导入完整镜像（推荐，与提交时环境逐字节一致）
#      4 个分卷在 04_镜像/ 下。可直接把拼接流喂给 docker import，省一份 12 GB 副本：
cat 04_镜像/hyperseg-rootfs-20260930.tar.gz.part_0* | docker import - hyperseg-uav:submit
#      若已按第五节合并出单文件，则：docker import hyperseg-rootfs-20260930.tar.gz hyperseg-uav:submit
docker run --gpus all -it --rm \
  -v /path/to/low_altitude_2026:/root/hyperseg/dataset/low_altitude_2026:ro \
  hyperseg-uav:submit bash
#   b) 用 Dockerfile 干净重建
cd 03_代码与复现 && docker/build.sh

# 数据：官方数据集需自行取得（见 06_数据说明/DATA_ACCESS.md），挂到
#   /root/hyperseg/dataset/low_altitude_2026/
cd /root/hyperseg

# 0) 让项目根可导入：镜像内那份 run_inference.py 早于 sys.path 自引导，
#    不设 PYTHONPATH 时 `import hyperseg_uav` 会报 ModuleNotFoundError
export PYTHONPATH=/root/hyperseg

# 1) 权重加载自检：必须打印 translated_legacy_encoder_keys: 628
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python experiment/b3_test2_20260929/run_inference.py --check-only

# 2) 复现有标注验证集上的 mIoU（期望 0.7420232）
python experiment/b3_test2_20260929/val_sanity.py

# 3) 重新导出测试集 2 预测，与 01_预测结果/ 内的 zip 比对像素
python experiment/b3_test2_20260929/run_inference.py \
    --input dataset/low_altitude_2026/test_2/images \
    --output /tmp/reproduce --checkpoint models/hyperseg_b3_best.pt
```

`03_代码与复现/scripts/01…05` 是等价的分步脚本（从 `03_代码与复现/` 调用，
脚本自己会 `cd` 进 `hyperseg/` 并设好 `PYTHONPATH`）；`docs/OPS.md` 给了完整命令，
`docs/REPRODUCE.md` 说明了**哪些结果可以精确复现、哪些不能**（推理可精确复现；
重训不能，因为检查点未记录原始训练参数）。

## 五、校验方式

```bash
# 1) 全包校验（含 4 个镜像分卷，约 12.2 GiB，需几分钟）
tr -d '\r' < CHECKSUMS.sha256 | sha256sum -c -

# 2) 只校验镜像目录（04_镜像/ 内另有一份同格式清单，用于单独交付镜像的场合）
cd 04_镜像 && tr -d '\r' < CHECKSUMS.sha256 | sha256sum -c -

# 3) 合并分卷并核对整体哈希：必须是 c541042f…a5c9d
cat hyperseg-rootfs-20260930.tar.gz.part_0* > hyperseg-rootfs-20260930.tar.gz
sha256sum hyperseg-rootfs-20260930.tar.gz
```

第 3 步会额外写出一个 12.18 GiB 的合并副本。**不落地也能验证**——直接对拼接流算哈希，
结果相同，且省下一份空间：

```bash
cat hyperseg-rootfs-20260930.tar.gz.part_0* | sha256sum   # 必须是 c541042f…a5c9d
```

`tr -d '\r'` 只是为兼容 Windows 上生成的 CRLF 行尾；若文件已是 LF，直接
`sha256sum -c CHECKSUMS.sha256` 效果相同。该写法对两种行尾都成立。

## 六、已知事项（如实说明）

1. **镜像已完成容器内冒烟（通过）**：2026-09-30 在真实 Docker 守护进程上
   `docker import` 成功（镜像 ID `9b65ec6f9121`，展开约 34.3 GB，`linux/amd64`），
   容器内 `run_inference.py --check-only` 打印出 `translated_legacy_encoder_keys: 628`、
   `epoch 141`、`val_miou 0.7420226665631505`、识别输入 **1300** 张。完整记录见
   `05_实验证据/logs/container_smoke_20260930.log`。**唯一未跑的是 `Dockerfile` 干净重建**
   （推荐路径是 `docker import`），提交物中没有结论依赖它。
2. **容器内需要 `PYTHONPATH=/root/hyperseg`**：镜像里那份 `run_inference.py`
   早于本次加入的 `sys.path` 自引导，不设该变量时 `import hyperseg_uav` 会报
   `ModuleNotFoundError`。提交包内的副本已修好，可裸跑；`scripts/0X_*.sh` 也会自动设置。
3. **重训不可精确复现**：检查点只存了 `model` / `model_config` / `epoch` / `val_miou`，
   没有记录产生它的训练参数。提交的预测可由所附权重复现，权重的来源（epoch、
   验证分数、结构、哈希）也完整记录，缺的是训练配方的来源。
4. 数据集**一个字节都没有进本包与镜像**，只保留了 `runs/splits/{train,val,test}.txt`
   这类文件名清单以保证划分可复现。
5. **镜像分卷共 12.18 GiB**：本目录是完整交付形态；若接收方有单文件体积限制，
   可只上传 `04_镜像/` 下的分卷并按 `README_IMAGE.md` 第二节合并（单卷最大 3.98 GiB）。
6. **MiT-B3 骨干随包交付**：`03_代码与复现/hyperseg/models/nvidia--mit-b3/`
   （`config.json` 70 045 B + `model.safetensors` 178 416 440 B，与镜像内逐字节一致）。
   `hyperseg_uav/model.py` 会优先取该本地目录，所以 **`Dockerfile` 重建不需要联网**
   下载骨干。推理本身只依赖 `hyperseg_b3_best.pt`（检查点内含完整编码器权重），
   骨干文件是给重训路径用的。
