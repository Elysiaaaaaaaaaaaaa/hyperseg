# 完整镜像文件 · 交付说明

本目录交付的是**复赛要求的「完整镜像文件」**：把运行本次提交的整台工作机
（AutoDL 容器 `server3`）的系统盘导出成根文件系统归档，再用 `docker import` 即可
在评审侧还原出一模一样的环境，进而在其中重跑本次提交的预测。

---

## 一、文件清单

| 文件 | 字节数 | SHA-256 |
| --- | --- | --- |
| `hyperseg-rootfs-20260930.tar.gz.part_01` | 3 984 588 800 | `dfd17b59b7fb6164b7f79c42c4726a4761447f21ba8d5d3b7a3ed342deedd3d2` |
| `hyperseg-rootfs-20260930.tar.gz.part_02` | 3 984 588 800 | `a0d14e54bb443c0d3289a59b70b89c0215c79f1cf18dbf7ed1c0ee8947b8229c` |
| `hyperseg-rootfs-20260930.tar.gz.part_03` | 3 984 588 800 | `38e9639123988fa217a4aec4ebf67149d1c217bb5806a0286563d8b1677ebc53` |
| `hyperseg-rootfs-20260930.tar.gz.part_04` | 1 125 952 193 | `db2662f43b881e4376fa211be801033faeb9e707d5b28da209b649554f60c878` |
| **合并后** `hyperseg-rootfs-20260930.tar.gz` | **13 079 718 593**（12.18 GiB） | **`c541042f509f4544a05a28a92054e714c152f8d9145597d76ee55551040a5c9d`** |
| `image_manifest.json` | 见文件 | 机器可读清单（含排除列表与关键文件哈希） |
| `CHECKSUMS.sha256` | 见文件 | 上述文件的 sha256，`sha256sum -c` 可校验 |

分卷原因：百度网盘对单文件大小有限制（普通用户 Windows/Mac 客户端 4 GB），
且我们的镜像导出为**单个 gzip 流**、不具备内部分片能力，所以按 **3800 MiB** 切卷，
每卷 3.71 GiB，对「4 GB 上限」留有安全余量。

## 二、还原镜像

```bash
# 1) 校验分卷完整
#    （tr 是为了兼容 Windows 上生成的 CRLF 行尾；本机若已是 LF，加不加都一样）
tr -d '\r' < CHECKSUMS.sha256 | sha256sum -c -

# 2) 合并分卷（顺序敏感：必须按 part_01 → part_04）
cat hyperseg-rootfs-20260930.tar.gz.part_01 \
    hyperseg-rootfs-20260930.tar.gz.part_02 \
    hyperseg-rootfs-20260930.tar.gz.part_03 \
    hyperseg-rootfs-20260930.tar.gz.part_04 \
  > hyperseg-rootfs-20260930.tar.gz

# 3) 核对合并结果，必须是
#    c541042f509f4544a05a28a92054e714c152f8d9145597d76ee55551040a5c9d
sha256sum hyperseg-rootfs-20260930.tar.gz

# 4) 导入为本地镜像
docker import hyperseg-rootfs-20260930.tar.gz hyperseg-uav:submit

# 5) 启动（本镜像不含数据集，数据集需另行申请并提供，见第六节）
docker run --gpus all -it --rm \
  -v /path/to/low_altitude_2026:/root/hyperseg/dataset/low_altitude_2026:ro \
  hyperseg-uav:submit bash
```

> **`docker import` 不是 `docker load`。** 本文件是容器**根文件系统快照**（gzip 压缩的
> tar），不是 `docker save` 生成的镜像归档，两者格式不同、不能混用。用 `docker load`
> 会直接报错。

## 三、镜像内容

| 项目 | 值 |
| --- | --- |
| 来源机器 | AutoDL 容器 `autodl-container-51bc499713-29f504bc`（本项目称 `server3`，GPU 为 RTX 4090） |
| 导出时刻 | 2026-09-30 01:08（UTC+8）／ 2026-09-29T17:08Z |
| 操作系统 | Ubuntu 22.04.3 LTS（jammy），`linux/amd64` |
| 格式 | gzip 压缩的根文件系统 tar（`tar --format=pax` + `gzip -3 -n`） |
| 工作副本 | `/root/hyperseg`（与提交包 `03_代码与复现/hyperseg` 同源） |
| Python 环境 | `/root/hyperseg/.venv-mask2former`（torch 2.3.0+cu121 / transformers 4.57.6）；另有 `/root/miniconda3` |
| 提交用权重 | `/root/hyperseg/models/hyperseg_b3_best.pt`（MiT-B3，epoch 141，记录 val mIoU 74.2023 %） |
| 骨干配置 | `/root/hyperseg/models/nvidia--mit-b3/` |
| 固定划分 | `/root/hyperseg/runs/splits/{train,val,test}.txt` |

**关键文件哈希**（导出时逐条校验，13/13 通过，全部 `critical ok`）：

| 容器内路径 | 字节数 | SHA-256 |
| --- | --- | --- |
| `/root/hyperseg/models/hyperseg_b3_best.pt` | 179 657 463 | `fb124302bafaec18aaa3f695dc021efead7bd00f710348d040888bb270148968` |
| `/root/hyperseg/models/nvidia--mit-b3/config.json` | 70 045 | `7a10fb664fa9f10a32361cb7191298e977b3f80a259b65682d2854c15bfc3345` |
| `/root/hyperseg/hyperseg_uav/model.py` | 10 403 | `5a69c25e317f9b480cd2addfc2c0d35b69597514cdef4a1d96efa689503d1d03` |
| `/root/hyperseg/hyperseg_uav/__init__.py` | 352 | `e0eb9325fd8e56e0f1fa84df345fbc8cf8783c44b51672f1388d33bb86293412` |
| `/root/hyperseg/tools/infer_hyperseg.py` | 3 276 | `08b892361940c8084a1bba9cc34b34db91bb88f4e9920df6c2d0576fd5f81654` |
| `/root/hyperseg/tools/test_hyperseg.py` | 3 587 | `1ff4c148b28afbbc074d5c5a5d9240d7c9310725bedc20c49943a6bd11fe0aee` |
| `/root/hyperseg/tools/check_submission.py` | 603 | `4b8246f9cf4d0a5a5f5fc5bdd299f1d1d19c054a0da20434566922beffb8f621` |
| `/root/hyperseg/runs/splits/train.txt` | 27 990 | `4d19ea1ece87be870f17fb04d760475f68a5344cfcb674386e1f37c5fc484622` |
| `/root/hyperseg/runs/splits/val.txt` | 3 495 | `39d349bffd31a2dd7bdccd03dac3225a5a30aa11c5e2c97fe7f0367a7ef940f8` |
| `/root/hyperseg/runs/splits/test.txt` | 3 495 | `09068e956dd175da0f417ba590844876ea542ec7f210261f5647272b34635e4f` |
| `/root/hyperseg/.venv-mask2former/pyvenv.cfg` | 228 | `eae8d62731aaaedd0ebc969835fc5ff9f317aedad06a1fa4f973046fe3ca2ead` |
| `/root/miniconda3/bin/python3.12` | 30 627 224 | `0c05a22b0b180580a76437114a95cf138f67c8f46245acad26017c803b42b8c1` |
| `/usr/bin/bash` | 1 396 520 | `2c336c63e26881d2f02f34379024e7c314bce572c08cbaa319bacbbec29f93ed` |

其中 `model.py` / `__init__.py` / `infer_hyperseg.py` 三者的哈希，正是**包含旧键名兼容修复**
（`legacy_encoder_key` / `translate_legacy_keys`）的版本——这也是为什么必须重新导出，
而不能沿用更早的 09-26 镜像：后者在 Transformers 4 环境下会因 628 个骨干键命名不一致
而无法加载提交权重。

## 四、镜像里**没有**什么（重要）

赛事数据集**一个字节都没有进镜像**，这是硬约束（手册禁止泄露、传播数据集）。
导出时被排除的路径：

```
/.dockerenv  /autodl-pub  /dev  /etc/{hostname,hosts,resolv.conf}  /init
/proc  /sys  /tmp/*  /run/*  /root/autodl-tmp  /root/hyperseg/dataset
```

因此：

- `/root/hyperseg/dataset` 在镜像中是**空的**；
- 镜像里保留的是**文件名清单**（`runs/splits/*.txt`），用于证明划分可复现；
  图像与标注本身不在包内；
- 导出脚本本身与流式传输助手放在排除区 `/root/autodl-tmp/`，不会出现在交付镜像里。

## 五、容器内验收（评审侧应做的三件事）

```bash
cd /root/hyperseg

# 0) 让项目根可导入。镜像内这份 run_inference.py 没有 sys.path 自引导，
#    不设 PYTHONPATH 时 `import hyperseg_uav` 会报 ModuleNotFoundError。
export PYTHONPATH=/root/hyperseg

# 1) 权重加载自检：必须打印 translated_legacy_encoder_keys: 628
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python experiment/b3_test2_20260929/run_inference.py --check-only

# 2) 复现有标注验证集上的 mIoU
#    期望 replayed 0.7420232149377345 vs recorded 0.7420226665631505（差 5.5e-7）
python experiment/b3_test2_20260929/val_sanity.py

# 3) 重新导出测试集 2 的预测，与 01_预测结果/predictions_test2_round2.zip 比对像素
python experiment/b3_test2_20260929/run_inference.py \
    --input dataset/low_altitude_2026/test_2/images \
    --output /tmp/reproduce --checkpoint models/hyperseg_b3_best.pt
```

上面第 0 步的等价「裸 docker」写法（本次实测用的就是这一条，数据集以只读挂载）：

```bash
docker run --rm \
  -v /path/to/low_altitude_2026/test_2/images:/root/hyperseg/dataset/low_altitude_2026/test_2/images:ro \
  -w /root/hyperseg -e PYTHONPATH=/root/hyperseg \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  hyperseg-uav:submit \
  /root/hyperseg/.venv-mask2former/bin/python \
    /root/hyperseg/experiment/b3_test2_20260929/run_inference.py --check-only
```

> 官方测试集在容器内的期望挂载路径是
> **`/root/hyperseg/dataset/low_altitude_2026/test_2/images`**
> （有标注的验证集为 `.../val/images` 与 `.../val/masks`）。镜像不含任何赛事图像，
> 需评审侧自行取得数据后挂载到该路径。

## 六、本次交付的验证状态（如实说明）

| 环节 | 状态 |
| --- | --- |
| 导出流完整性（`gzip -t`） | ✅ 通过 |
| 归档内 13 个关键文件逐条哈希比对 | ✅ 13/13 `critical ok` |
| 归档内 `dataset/` 条目数 | ✅ 0（数据集确已排除） |
| 本机副本 SHA-256 复算 | ✅ 与导出记录一致（`c541042f…`） |
| 分卷拼接无损（`cat part_* \| sha256sum`） | ✅ 等于 `c541042f…` |
| **容器内 `docker import` + 冒烟** | ✅ **通过**（2026-09-30 实跑） |

容器内冒烟的具体结论：

| 项 | 值 |
| --- | --- |
| 导入命令 | `docker import server3-system-no-dataset-20260930.rootfs.tar.gz hyperseg-uav:submit` |
| 镜像 ID | `sha256:9b65ec6f91219c9406ebfe6b23a74ec00d15dfa41c9362481916cef55585fca0` |
| 解压后大小 | 13 079 719 289 字节（约 34.3 GB 展开） |
| 平台 | `linux/amd64` |
| 容器内自检 | `translated_legacy_encoder_keys: 628`；严格加载成功；`epoch 141`；`val_miou 0.7420226665631505`；识别输入 **1300** 张、边长 1024 |
| 容器内 Python 环境 | `python 3.12`、`torch 2.3.0+cu121`、`cuda_available: false`（本次冒烟未用 GPU，仅验证加载与清单） |
| 完整记录 | `05_实验证据/logs/container_smoke_20260930.log` |

**仍未执行的一项**：`Dockerfile` 的干净重建（第四节第 1 步的 b 方案）没有在干净守护进程里跑过；
本次验证走的是 `docker import` 这条（也就是推荐路径）。提交物中没有任何结论依赖 Dockerfile 重建。

## 七、附：本卷的生成方式

镜像由仓库内脚本导出与下载，评审如需自行复现该过程：

```bash
# 服务器侧：导出根文件系统（自动排除数据集与挂载点，内部已 tar | gzip -3 -n）
python tools/export_rootfs.py export --server server3 > rootfs.tar.gz

# 本地侧：流式下载 + 边下边校验（逐条核对关键文件哈希、拒绝排除列表中的路径）
python tools/download_rootfs.py --server server3 \
    --label server3-system-no-dataset-20260930 --out <目标目录>

# 切分卷（3800 MiB）
split -b 3800M -d --numeric-suffixes=1 --suffix-length=2 \
      hyperseg-rootfs-20260930.tar.gz hyperseg-rootfs-20260930.tar.gz.part_
```

两个脚本一并放在提交包 `03_代码与复现/image-tools/` 下。

导出脚本内部已做 `tar | gzip -3 -n`，因此**输出本身就是 gzip 流**，请勿再套一层压缩。
