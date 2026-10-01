# HyperSeg-UAV v2（resume 检查点）复赛测试集 2 推理

在 **server2**（AutoDL，RTX 4090 24 GiB）上用 `models/hyperseg_resume_best.pt` 对复赛
无标注测试集 `low_altitude_2026/test_2/images`（1300 张 1024×1024 PNG）做滑窗推理，
并把预测打包下载回仓库。

**2026-09-30 已完成。** 提交文件为
[hyperseg_b3_resume_test2_predictions.zip](hyperseg_b3_resume_test2_predictions.zip)
（1300 张预测，12 085 143 字节 ≈ 11.53 MiB），服务器提交检查、本地 ZIP 复核和
手册第十三节格式复核均通过。

## 检查点

| 项 | 值 |
| --- | --- |
| 文件 | `models/hyperseg_resume_best.pt` |
| SHA-256 | `026b21d99ee6befc357b3e93fe78adefc4dfd457e249d1ff69e5bc581b2b3b48` |
| 结构 | **HyperSeg-UAV v2**（`tools/model_1.py`），MiT-B3 骨干，9 类，`version="v2"` |
| 状态字典 | 730 个张量（v1 为 686） |
| 训练轮次 | epoch 134 |
| 记录 Val mIoU | 76.3160%（768×768，固定 `runs/splits/val.txt`，batch 8） |

### 这个检查点不是 v1，不能走 `tools/infer_hyperseg.py`

`models/hyperseg_b3_best.pt`（上次提交用的）是 `hyperseg_uav/model.py` 的 v1 架构。
`hyperseg_resume_best.pt` 是 **v2**，两者参数集不同：

| 前缀 | v1 张量数 | v2 张量数 |
| --- | --- | --- |
| `encoder` | 628 | 628 |
| `fusion` | 12 | 40 |
| `low_rank` | 2 | 6 |
| `boundary_refine` | — | 12 |
| 其余（scene / modulation / decoder / head / boundary） | 44 | 44 |
| 合计 | 686 | 730 |

v2 的构造器还多一个 `version` 字段，因此 `HyperSegUAV(**config)` 会直接抛
`unexpected keyword argument 'version'`；必须先 `config.pop("version")` 再用
`tools/model_1.py` 的类构造，然后逐键映射（`key if key in destination else legacy_encoder_key(key)`）。
这套加载逻辑封装在 [v2_model.py](v2_model.py)，且**双向严格**（730 个张量全部落位，
缺任何一个都报错）。

编码器键名与 v1 一样是 Transformers 5 的 `encoder.backbone.stages.{i}.blocks.{j}...`，
需要转换 **628** 条键；`hyperseg_uav.model.legacy_encoder_key` 与
`experiment/loveda_fewshot/train.py` 里的同名函数逐字节等价。

## 推理协议：768 / 0.5 / 无 TTA（经过标定，不是沿用默认）

v2 的训练脚本不在仓库里，`val_miou` 也没有记下分辨率，所以协议是**实测标定**出来的
（见 [experiment.md](experiment.md) 与 `logs/v2_*.json`）：

| 组合 | 重放 Val mIoU | 与记录差 |
| --- | --- | --- |
| size 768 / batch 2 / 无归一化 | 0.740312 | 2.28e-02 |
| size 512 / batch 2 / 无归一化 | 0.704259 | 5.89e-02 |
| size 1024 / batch 2 / 无归一化 | 0.742605 | 2.06e-02 |
| size 768 / ImageNet 归一化 | 0.718650 | 4.45e-02 |
| **size 768 / batch 8 / 无归一化** | **0.7631586** | **9.28e-07** ✅ |

结论：预处理与 v1 相同（只做 `/255`，**不加** ImageNet 归一化），窗口 **768**，
评估 batch 为 8，**不加 TTA**。因此导出协议与本项目既有提交完全一致，两者可直接比较。
在 batch 8 上复现到 1e-6 也确认了权重被完整、正确地还原。

## 路径与设置

| 项目 | server2 路径或设置 |
| --- | --- |
| 输入图像 | `/root/hyperseg/dataset/low_altitude_2026/test_2/images` |
| 检查点 | `/root/hyperseg/models/hyperseg_resume_best.pt` |
| 工作目录 | `/root/hyperseg/runs/b3_resume_test2_20260930` |
| 推理设置 | FP32，768×768 滑窗，50% 重叠，无 TTA |
| 每张图的窗口数 | 4（stride 384，覆盖 1024×1024） |
| 输出 | 工作目录下 `predictions/` 与 `hyperseg_b3_resume_test2_predictions.zip` |

## 为跑通这次推理对 server2 做的改动

server2 的系统盘代码是 **09-26 快照**，缺 09-29 的键名修复，也缺 v2 的模型定义。
已同步下列文件（原始版本备份在 `/root/autodl-tmp/backup_b3_resume_20260930/`）：

| 文件 | 原 SHA-256（已备份） | 现 SHA-256 |
| --- | --- | --- |
| `hyperseg_uav/__init__.py` | `b527a353…` | `e0eb9325…` |
| `hyperseg_uav/model.py` | `d14f65a4…` | `5a69c25e…` |
| `tools/infer_hyperseg.py` | `a952afde…` | `08b89236…` |
| `tools/model_1.py` | 服务器上原本没有 | `24ab2698…` |

这些改动只做向后兼容的键名转换与新增文件，不改变既有语义；同步后本仓库
`hyperseg_uav/data.py`、`losses.py`、`swin_l_model.py` 与服务器仍逐字节一致。

## 运行命令

```bash
cd /root/hyperseg
# 预检：1300 张 1024×1024、检查点 9 类、严格加载（应打印 translated_legacy_encoder_keys: 628）
.venv-mask2former/bin/python -u experiment/b3_resume_test2_20260930/infer_v2_test2.py --check-only --size 768
# 导出（后台，写 inference.pid / runner.log，已有预测时拒绝重复启动）
bash experiment/b3_resume_test2_20260930/launch_server2.sh 768
```

`infer_v2_test2.py` 内联实现滑窗（逻辑逐行抄自 `tools/infer_hyperseg.py`，保证可比），
结束后自动执行 `tools/check_submission.py`、打包 ZIP 并写带 SHA-256 的 `result.json`；
`exit_code=0` 表示完整成功。**本次实测 201.3 秒**（约 6.5 张/秒）。

协议标定与权重复核（跑在 GPU 上，约 2 分钟）：

```bash
.venv-mask2former/bin/python -u experiment/b3_resume_test2_20260930/v2_protocol_search.py --sizes 768
```

本地复核下载到的 ZIP：

```bash
python experiment/b3_resume_test2_20260930/verify_download.py        # 条数/尺寸/模式/类别/文件名
python experiment/b3_resume_test2_20260930/check_manual_compliance.py # 手册第十三节，解析 PNG 块
python experiment/b3_resume_test2_20260930/compare_v1_v2.py          # 与既有 v1 提交的差异
```

## 结果与回收

- ZIP SHA-256：`07d3b396ad45c11bb8d587d05823a35944e7d3bf9868998d832c92ce2db31e1e`，
  与服务器 `result.json` 记录一致（下载后本地重算）。
- 手册复核判定 **PASS**：1300 个条目、CRC 通过、无重复、无嵌套路径、
  全部 IHDR 颜色类型 0（灰度）、文件名与官方测试集 2 逐一对应。
- 预测类别份额：Background 33.83%、Vegetation 21.07%、Building 18.04%、
  Agricultural 10.82%、Water 7.74%、Road 6.89%、Barren 1.01%、Vehicle 0.60%。
- 与 v1 提交相比：**8.90% 的像素被改判**，1295/1300 张有变化；最明显的是
  Barren 类份额降到 v1 的 0.696 倍，Water 升到 1.054、Agricultural 升到 1.050。
- `logs/` 收录 `runner.log`、`preflight.json`、`result.json`、`submission_check.log`、
  `exit_code`、`v2_protocol_search.{json,log}`、`v2_metric_probe.json`、
  `v2_resolution_replay.json`、`v2_norm_grid.log`、`v1_v2_comparison.json`。
- 需要重跑时先删除服务器工作目录下的 `predictions/`，否则启动器会拒绝启动。

## 口径限制

无标注测试集没有真值，**不能报告 mIoU**。76.3160% 是检查点在
`runs/splits/val.txt`（699 张有标注图）上的历史记录值，本次只是把它复现出来用于
确认权重与协议，不是本预测包的成绩。
