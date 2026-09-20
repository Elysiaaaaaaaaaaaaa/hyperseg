# SUIM 固定协议 Few-shot 实验

本实验把 9 类 HyperSeg-UAV v2 checkpoint 迁移到 SUIM 的 8 个语义类别。训练使用完整语义 mask、同一套嵌套支持集和固定更新步数，最终在官方 110 张 `TEST` 上按原始分辨率评估。

SUIM 类别 `0` 是有效类别 `BW_background_waterbody`，训练和评估均不能忽略。RGB mask 按三个二进制位解码：

```text
class_id = (R >= 128) * 4 + (G >= 128) * 2 + (B >= 128)
```

读取器会处理已知的尺寸异常：当 `train_val` mask 比对应图像恰好多 55 行时，只裁掉 mask 底部；其他尺寸不一致直接报错。图像缩放使用双线性插值，mask 使用最近邻插值，训练缩放始终保持原宽高比。

## 实验矩阵

固定矩阵包含四种“更新范围 × 训练预算”组合：

| setting | 可训练模块 | steps |
|---|---|---:|
| `adapter_200` | scene、modulation、fusion、low-rank adapter、decoder、boundary refinement 和 heads | 200 |
| `adapter_2000` | 同上 | 2000 |
| `semantic_head_200` | 只训练语义分类头 `model.head` | 200 |
| `semantic_head_2000` | 只训练语义分类头 `model.head` | 2000 |

默认使用 K=`1,2,5,10` 和 seed=`3407,3408,3409`，共 48 次运行。所有组合使用完全相同的逐类别有序支持清单。同一图像可以服务多个类别，因此实际训练图数可能少于 `8*K`。

主实验严格迁移 `models/hyperseg_resume_best.pt` 中所有类别无关参数，并随机初始化 SUIM 的 8 通道语义头。`--head-init semantic-map` 是可选消融，只复制 `Water -> BW`、`Vegetation -> PF`、`Building -> WR`、`Barren -> SR`，其余四个目标通道仍为随机初始化。随机 head 没有可解释的 0-shot 类别对应关系，因此实验不设置 0-shot。

## 1. 生成并审计固定协议

先在用于生成协议的环境中安装项目依赖，然后从项目根目录运行：

```bash
python -u experiment/fewShot_SUIM/prepare_protocol.py \
  --data-root SUIM \
  --output-dir runs/suim_fewshot_protocol_seed3407 \
  --selection-seed 3407
```

脚本读取全部 1525 张训练 mask，把颜色和尺寸异常写入 `dataset_audit.json`，并生成 `selection.json`、`1shot.json`、`2shot.json`、`5shot.json`、`10shot.json` 和 `evaluation.json`。已有协议不会被覆盖。

自动选择可复现，默认要求候选图中目标类别至少有 512 像素、占整图至少 0.05%。如果要使用人工复核的支持集，可传入每类恰好十张有序图像的 JSON：

```json
{
  "per_class": {
    "0": ["image_0.jpg", "... 另九张 ..."],
    "1": ["image_1.jpg", "... 另九张 ..."],
    "2": ["image_2.jpg", "... 另九张 ..."],
    "3": ["image_3.jpg", "... 另九张 ..."],
    "4": ["image_4.jpg", "... 另九张 ..."],
    "5": ["image_5.jpg", "... 另九张 ..."],
    "6": ["image_6.jpg", "... 另九张 ..."],
    "7": ["image_7.jpg", "... 另九张 ..."]
  }
}
```

生成协议时增加 `--selection-file path/to/selection.json`。脚本会检查每张图确实含有对应目标类别。

也可以直接使用带叠加预览的 GUI 逐类筛选。它读取 `train_val` 的 JPG/RGB-BMP
标注，选择记录会自动保存到 `runs/suim_manual/selection.json`，导出按钮还会生成
包含官方 `TEST` 评估清单的完整协议目录：

```bash
python SUIM/select_suim_gui.py \
  --data-root SUIM \
  --output runs/suim_manual
```

如果只想把 GUI 选择结果交给协议生成脚本：

```bash
python -u experiment/fewShot_SUIM/prepare_protocol.py \
  --data-root SUIM \
  --output-dir runs/suim_fewshot_manual \
  --selection-file runs/suim_manual/selection.json
```

无需加载模型即可检查协议和文件：

```bash
python experiment/fewShot_SUIM/run.py \
  --data-root SUIM \
  --manifest-dir runs/suim_fewshot_protocol_seed3407 \
  --output-dir runs/check-unused \
  --shots 1 --mode adapter --steps 200 --check-only
```

## 2. 检查和运行实验矩阵

先查看即将启动的完整命令：

```bash
python experiment/fewShot_SUIM/run_grid.py \
  --data-root SUIM \
  --manifest-dir runs/suim_fewshot_protocol_seed3407 \
  --dry-run
```

建议先在服务器用单 seed、K=`1,2` 跑通 8 次 pilot：

```bash
python -u experiment/fewShot_SUIM/run_grid.py \
  --data-root /root/autodl-tmp/SUIM \
  --manifest-dir /root/autodl-tmp/hyperseg/runs/suim_fewshot_protocol_seed3407 \
  --init-checkpoint /root/autodl-tmp/hyperseg/models/hyperseg_resume_best.pt \
  --output-root /root/autodl-tmp/hyperseg/runs/suim_fewshot_pilot \
  --shots 1 2 --seeds 3407 --amp --device cuda
```

服务器必须能离线加载 `nvidia/mit-b3`。如果 Hugging Face 缓存和项目内 `models/nvidia--mit-b3/` 都不存在，在上述命令中增加 `--backbone-path /path/to/mit-b3`。

pilot 正常后运行完整 48 次实验：

```bash
python -u experiment/fewShot_SUIM/run_grid.py \
  --data-root /root/autodl-tmp/SUIM \
  --manifest-dir /root/autodl-tmp/hyperseg/runs/suim_fewshot_protocol_seed3407 \
  --init-checkpoint /root/autodl-tmp/hyperseg/models/hyperseg_resume_best.pt \
  --output-root /root/autodl-tmp/hyperseg/runs/suim_fewshot_v1 \
  --shots 1 2 5 10 --seeds 3407 3408 3409 \
  --settings adapter_200 adapter_2000 semantic_head_200 semantic_head_2000 \
  --amp --device cuda
```

中断后可加 `--skip-completed` 续跑。只有签名完全一致的已完成目录会被复用；配置变化或未完成目录不会被覆盖。

## 输出与指标

每次运行输出：

- `selection.json`：本次使用的准确支持清单；
- `run_config.json`：checkpoint/协议哈希、可训练参数名、类别权重和迁移报告；
- `history.jsonl`：每个 optimizer step 的训练损失；
- `final.pt`：固定最终步 checkpoint；
- `summary.json`：官方 TEST 指标。

矩阵根目录会在每次完成后更新 `results.csv` 和 `aggregate.json`。主指标是原尺寸 8 类 mIoU，同时记录逐类 IoU、排除 BW 的 foreground mIoU、像素准确率和完整 8×8 混淆矩阵。TEST 不参与 checkpoint 选择。

本地轻量检查：

```bash
python -m unittest discover -s experiment/fewShot_SUIM -p "test_*.py"
python -m py_compile experiment/fewShot_SUIM/*.py
```
