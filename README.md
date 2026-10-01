# HyperSeg-UAV · 无人机低空航拍图像语义分割

基于 PyTorch 的无人机低空航拍图像语义分割项目。输入 RGB 图像，输出 9 类像素级预测
（类别 `0` 为 Ignore，不参与损失与 mIoU）。主模型 **HyperSeg-UAV** 以 SegFormer **MiT-B3**
为骨干，叠加场景条件动态调制、多尺度动态融合与低秩适配器，配语义 + 边界双预测头。

- 代码讲解（逐模块、以代码现状为准）：[`HYPERSEG.md`](HYPERSEG.md)
- 协作约定与数据集路径对照表：[`AGENTS.md`](AGENTS.md)
- 实验总记录：[`experiment.md`](experiment.md)（分实验细节见 `experiment/<实验名>/experiment.md`）
- 复赛提交包说明：[`packaging/README.md`](packaging/README.md)

---

## 1. 任务与类别

| ID | 类别 | ID | 类别 |
|---:|---|---:|---|
| 0 | Ignore（不参与损失/mIoU） | 5 | Barren |
| 1 | Background | 6 | Vegetation |
| 2 | Building | 7 | Agricultural |
| 3 | Road | 8 | Vehicle |
| 4 | Water | | |

标签定义同时写在 `dataset/low_altitude_2026/Label.txt`。训练/评估默认 `classes=9`、
`ignore_index=0`，有效类别为 `1..8`。

## 2. 数据集

主数据集 `low_altitude_2026`，本机根目录 `dataset/low_altitude_2026/`：

| 用途 | 路径 | 规模 |
| --- | --- | ---: |
| 有标注训练图像 / 掩码 | `train/images`、`train/masks` | 6996 / 6996（单通道，像素 `0..8`） |
| 固定训练划分 | `runs/splits/train.txt` | 5598 |
| 固定验证划分 | `runs/splits/val.txt` | 699 |
| 固定有标注测试划分 | `runs/splits/test.txt` | 699 |
| 无标注比赛测试集（初赛） | `images/` | 500 |
| 无标注比赛测试集（复赛） | `test_2/images/` | 1300 |

两个 "test" 必须区分：`runs/splits/test.txt` 是从有标注数据里划出的**离线评估集**，可以算 mIoU；
`images/` 与 `test_2/images/` 是**无标注比赛测试集**，只能导出预测、不能算指标。

划分文件每行是不带 `.png` 的 stem（如 `test1_123`），图像与掩码文件名必须严格对应。
图像缩放用双线性、掩码缩放用最近邻，保证空间对齐且不产生非法类别编号。

## 3. 目录结构

```text
hyperseg/
├── hyperseg_uav/          # 主模型：data.py / model.py / losses.py / swin_l_model.py
├── fewshot_hyperseg/      # episodic few-shot + HyperNetwork 模型（独立分支，见 HYPERSEG.md §11）
├── tools/                 # 训练、测试、滑窗推理、可视化、提交检查、打包与镜像工具
├── experiment/            # 每个实验一个独立目录（含 experiment.md 研究日志与 logs/）
├── runs/splits/           # train/val/test 固定划分
├── models/                # 权重（b3 v1 / v2）+ 本地 MiT-B3 骨干 models/nvidia--mit-b3/
├── dataset/               # 数据（不入版本控制）
├── packaging/             # 提交包手工文件权威副本（README/Dockerfile/configs/scripts/docs）
├── doc/                   # 技术报告、提交物组织规范
├── dist/                  # 组装后的交付目录（gitignore）
└── AGENTS.md / HYPERSEG.md / experiment.md / README.md
```

`experiment/` 现有实验：

| 目录 | 内容 |
| --- | --- |
| `b3_test2_20260929` | MiT-B3 复赛 test_2 推理、镜像冒烟、手动合规自检 |
| `b3_resume_test2_20260930` | v2 权重（`hyperseg_resume_best.pt`）加载与 test_2 复核 |
| `mask2former_uav` | H2 Swin-L + Mask2Former、H3 Swin-L HyperSeg、跨协议对比与塌陷诊断 |
| `h3_bgfix_20260930` | 无标注集背景塌陷修复（决策层惩罚 / 数据增强）消融 |
| `mathseg_uav` | MathSeg 数学解析分支与 few-shot 网格实验 |
| `fewShot_SUIM`、`loveda_fewshot` | 跨数据集 few-shot 微调（LoveDA / SUIM） |
| `docker_validation_20260927` | 镜像导入与容器内验证 |

## 4. 环境

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install -r requirements.txt   # torch>=2.2 / torchvision / Pillow / numpy / transformers>=4.40,<5
```

- 骨干默认走 Hugging Face `nvidia/mit-b3`；本地已放置 `models/nvidia--mit-b3/`，
  构造模型时优先取该目录，因此**离线也能建模型**。
- 本地算力不足，长训练须放服务器跑，用后台方式启动并把日志写入文件，结束后回收到对应
  `experiment/<实验名>/logs/`。

## 5. 快速开始

从项目根目录运行：

```bash
# 训练（验证图像会被缩放到 size×size）
python tools/train_hyperseg.py --data dataset --split-dir runs/splits \
    --out runs/hyperseg_best.pt --epochs 60 --size 768 --batch-size 2 --amp

# 在固定有标注测试划分上评估，写 <ckpt>_test_metrics.json
python tools/test_hyperseg.py --checkpoint models/hyperseg_b3_best.pt --split-dir runs/splits --size 768

# 滑窗推理并导出掩码（无标注测试集）
python tools/infer_hyperseg.py --checkpoint models/hyperseg_b3_best.pt \
    --input dataset/low_altitude_2026/test_2/images --output runs/predictions \
    --size 768 --overlap 0.5

# 可视化 + 提交格式自检
python tools/visualize_predictions.py --predictions runs/predictions \
    --images dataset/low_altitude_2026/test_2/images --output runs/visualized
python tools/check_submission.py runs/predictions dataset/low_altitude_2026/test_2/images
```

默认超参：60 epoch、crop 768、batch 2、lr 1e-4、AdamW（编码器 `lr*0.1`）、梯度裁剪 1.0、种子 3407。
训练只在验证 mIoU 刷新历史最佳时保存 checkpoint（`model` / `model_config` / `epoch` / `val_miou`）。

few-shot 分支的入口是 `tools/train_fewshot.py`，与单图模型互相独立，推理时必须提供
support 图像与 support 掩码，因此**不能**用 `infer_hyperseg.py` 加载其 checkpoint。

## 6. 权重与推理协议

| 权重 | 结构 | 规模 | 记录指标 |
| --- | --- | --- | --- |
| `models/hyperseg_b3_best.pt` | HyperSegUAV + MiT-B3（v1） | 179 657 463 B，686 张量 | epoch 141，val mIoU **74.2023 %** |
| `models/hyperseg_resume_best.pt` | `tools/model_1.py` 定义（**v2**，非 v1 续训版） | 540 532 365 B，**730 张量** | epoch 134，val mIoU **76.3160 %** |

**加载注意**

- `hyperseg_b3_best.pt` 由 Transformers 5 保存，骨干键名为 `encoder.backbone.stages.{i}...`，
  在 Transformers 4 环境必须经 `hyperseg_uav.translate_legacy_keys()` 转换（**628 个键**）。
  `tools/infer_hyperseg.py` 与 `tools/test_hyperseg.py` 已内置转换；后者用 `strict=False`，
  **日志里出现 `missing=628` 就是没转换、加载失败**。
- v2 需要带 `version` 的配置构造，用 v1 类会报 `unexpected keyword argument 'version'`；
  现成加载器见 `experiment/b3_resume_test2_20260930/v2_model.py`。
- **复核 val mIoU 必须同时复现口径**：v2 的 76.316 % 是 `size 768 / 评估 batch 8 / 无 TTA / 只做 /255`；
  换成 v1 的 batch 2 会得到 74.0312 %。对不上先怀疑 batch size。
- **推理协议不要擅自改**：MiT-B3 系 768 滑窗 / 0.5 重叠 / **无 TTA**；H3 Swin-L 用 512；
  MMSeg 系走整图 1024。`--tta` 目前只是水平翻转平均，不是多模型集成。

## 7. 提交与交付

比赛队名 `Mondstadt`。复赛提交 = 完整镜像 + 技术方案 + 网盘链接，组织规则见 `doc/提交物组织规范.md`。

- 正式镜像 13 079 718 593 B（12.18 GiB），sha256 `c541042f…a5c9d`，切 4 卷（每卷 3800 MiB）。
- 完整提交包目录：`dist/低空航拍语义分割_复赛_Mondstadt_20260929_完整交付/`（82 文件 / 12.53 GiB）。
- 打包工具：`tools/build_submission_package.py --root <包>`（幂等重建）与
  `tools/assemble_delivery_tree.py --package --volumes --out <dir>`（组装完整交付树）。
  **`dist/` 被 gitignore，手工文件的权威副本在 `packaging/`，只改 `dist/` 会丢。**
- 预测包自检不能只信 Pillow `mode=='L'`，必须解析 PNG 块确认 IHDR 颜色类型为 0、无 PLTE/tRNS
  （`experiment/b3_test2_20260929/check_manual_compliance.py`）。

## 8. 常见坑速查

- 加载 v1 权重必须先转换 628 个 legacy 键；看到 `missing=628` 即失败。
- 评估指标对 batch size 敏感，复现数字要连口径一起对齐。
- MMSeg `work_dir` 里没有 `train.log`，日志在 `<work_dir>/<时间戳>/<时间戳>.log`，
  验证行的键是裸 `mIoU`（不是 `val/mIoU`）。
- MMSeg 导出的预测 id 是原始标签空间 `1..8`（**1 = Background**），出现 `0` 才是 ignore 泄漏。
- 无标注测试集存在"低细节 × 蓝绿偏色"导致的背景塌陷，`test.txt` 变好**不能**推断 `test_2` 变好。
