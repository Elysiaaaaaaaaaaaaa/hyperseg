# H3-BgFix：针对测试集背景塌陷的训练策略改造

本实验在 **Swin-L 编码器 + 仓库 HyperSeg 解码器（即 H3）** 上，验证一套针对
`test_2` 上「大量像素被预测为背景」的训练策略改造。

诊断依据与全部原始证据见
[`_test2_extract/`](../../_test2_extract/) 与工作记忆
`D:/myproject/hyperseg/.workbuddy/memory/2026-09-30.md`。

## 1. 要解决的问题（先说清楚，否则改不对）

把 `test_2` 1300 张按「预测背景占比」分桶后，**其余每一类都单调衰减**，不是某一类映射错了：

| 分桶 | n | Bg | Bld | Road | Water | Veg | Agri |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 训练真值先验 | – | 27.78 | 18.04 | 8.88 | 6.12 | 28.34 | 6.86 |
| bg<0.2 | 458 | 9.88 | 24.57 | 10.09 | 9.38 | 32.27 | 12.77 |
| bg 0.4–0.6 | 206 | 48.62 | 15.08 | 5.79 | 5.95 | 12.27 | 9.16 |
| bg>0.8 | 100 | 90.72 | 2.31 | 0.44 | 1.71 | 2.13 | 2.42 |

MiT-B3 与 Swin-L 在**同一批图**上以相关系数 0.876 同幅失效，所以这不是架构问题。
塌陷子集的图像统计与训练集对比：R/G/B 由 106/108/100 变为 81/90/91（G−R 由 +1.4 变为 +8.3，
即更暗、青绿偏色），暗像素占比 19%→32%，平均梯度（细节尺度）6.09→3.68。

**同时必须避免过度修正**：查过训练真值后确认，类别 1 Background 在这套标注里是**真实兜底类**
（公路旁的非道路硬化面、塘埂、裸铺装都标 1），所以城市广场、立交桥那批图的"背景"预测
**可能本来就对**。因此本实验**不做全局背景降权**，改用只在 `GT≠1` 处生效的定向惩罚。

## 2. 改造内容与对照设计

每一处改造都是**独立开关**，preset 只负责给默认值，可以直接用命令行覆盖做单变量归因。

| preset | 增强 | 假背景惩罚 | 类别权重 | 选点指标 |
| --- | --- | --- | --- | --- |
| `baseline` | legacy（原 `UAVDataset`） | 0.00 | none | 原生 mIoU |
| `loss` | legacy | 0.25 | mild | 原生 mIoU |
| `aug` | bgfix | 0.00 | none | 原生 mIoU |
| `full` | bgfix | 0.25 | mild | 原生 + 代理 |

`baseline` 直接用仓库原有的 `hyperseg_uav.UAVDataset` 与 `hyperseg_loss`，**不经过任何重写**，
因此它不可能因为复现偏差而与已发布的 H3 结果漂移。

### 2.1 增强改造（`bgfix_dataset.py`）

| 项 | 原实现 | 本实验 | 针对的证据 |
| --- | --- | --- | --- |
| 缩放 | `scale∈{0.5..1.5}` 但被 `max(size, W*scale)` 钳回，**降采样分支失效**；且 `nw==size` 时随机裁剪退化为整图 | 保留钳位的 legacy 分支，同时新增 **真 zoom-out**：缩不下就整幅贴进 `size×size` 画布（图像填 0、掩码填 ignore） | 塌陷子集细节尺度只有训练的 ~0.6 |
| 有效分辨率 | 无 | **GSD 随机化**：以 `p=0.5` 把 patch 缩到 `r∈[0.5,1.0]` 再放大回 `size`（掩码不动） | 平均梯度 6.09→3.68 |
| 光度 | 只有 Brightness/Contrast + Color | **光度域随机化**：定向色偏（方向由实测推导）、逐通道白平衡、曝光、gamma、雾/低对比、Gaussian blur、噪声 | 更暗 + 青绿偏色 + 低对比 |
| 场景裁剪 | `scene_crop_prob=0.3`，稀有类 (5,7,8) | `0.5`，稀有类扩到 (3,4,5,7,8) | Road/Water 也被吞 |

定向色偏的方向**不是猜的**，而是从实测反推（见 `bgfix_dataset.py` 常量注释）：
`1 - 81.4/106.3 = 0.2342`、`1 - 89.7/107.8 = 0.1679`、`1 - 91.3/100.2 = 0.0888`；
`cast_strength=1.0` 时 amount=1.0 恰好复现实测偏色，更小的 amount 向内插值回训练观感。
（初版手选的 `(0.30,0.20,0.05)` 够不到实测幅度，被冒烟测试抓出来了。）

### 2.2 损失改造（`bgfix_loss.py`）

保留原权重结构（0.55 focal CE + 0.30 mean-class Dice + 0.025 rare + 0.10 boundary），新增一项：

```text
false_background = mean over {valid pixels with target != 1} of  -log(1 - p_bg)
```

外加把**从未被传入**的 `class_weights` 真正接上，默认 `mild`：
`(median_freq / freq) ** 0.5` 并夹到 `[0.7, 3.0]`，背景取 `max(w, floor)` —— 只轻微再平衡，
不压制任一多数类。`median` 模式留给更激进的消融。

### 2.3 选点与验证协议

- 原生验证：whole-image、原生分辨率、batch 1、全局 9×9 混淆矩阵 —— 与已发布 H3 选择检查点的协议一致。
- **代理验证**：把有标注 val 图渲染到 `0.5 / 0.625 / 0.75` 倍细节尺度，在降采样尺度上评 mIoU。
  这是**唯一能用有标注数据复现的 `test_2` 漂移轴**。
- 选点分：`(1-w) * 原生 mIoU + w * 代理 mIoU`，`w=0.3`（原生仍是主指标）。
- 同步记录 **背景假阳性率 / 假阴性率**，两者一起看才能发现过度修正。

**代理的必要性而非充分性**：它只模拟细节尺度，不模拟色偏、雾和新增场景类型。
代理上升只能说明「细节尺度这一侧的裕度变好了」，不能直接换算成比赛分数。

## 3. 文件清单

| 文件 | 作用 |
| --- | --- |
| `bgfix_dataset.py` | 增强改造数据集 + 代理评估路径 |
| `bgfix_loss.py` | 假背景惩罚、类别权重、加权指标 |
| `train_h3_bgfix.py` | 训练入口（preset 开关、代理验证、综合选点） |
| `eval_proxy.py` | 对检查点做原生 + 多尺度代理评估，支持多检查点并排 |
| `regression_test2.py` | `test_2` 无标注回归：类别占比 vs 训练先验、背景分布、分桶衰减、与旧预测的 delta |
| `smoke_test.py` | 上服务器前的自检：空间对齐、zoom-out 分支、GSD/光度、损失各项、指标 |
| `launch_h3_bgfix.sh` | 后台启动器（flock、CUDA 断言、先跑冒烟再起训练、退出码落盘） |
| `screen_parallel_h3_bgfix.sh` | 带槽位限制的并行调度器：跑完一臂再补一臂，失败即中止 |
| `screen_h3_bgfix.sh` | 串行版调度器（不需要并发时用） |
| `summarize_screen.py` | 拉回各臂 `val_history.jsonl`，出对照表与逐臂最优 |
| `watch_screen.py` | **本地**守候训练进度，全部完成后自动汇总（关掉笔记本就没了） |
| `watch_remote.sh` | **服务器侧**守候：把进度落到 `runs/mask2former_uav/h3_bgfix_watch.log`，事后下载 |
| `eta_screen.py` | 完成时间估算：快照差分测有效速率 + 槽位队列模拟排队臂 |
| `remote.py` | paramiko 远程通道：`check` / `sync` / `put` / `run` / `pull` / `collect` / `tail` |
| `configs/train_class_pixel_counts.json` | 训练划分的真实类别像素统计（5598 张，58.7 亿像素） |
| `results/screen_20k/` | `summarize_screen.py` 缓存下来的各臂原始曲线 |

**本实验不改动仓库任何既有文件**，全部是 `experiment/h3_bgfix_20260930/` 下的新增文件。

## 4. 运行方式

### 4.1 本地自检（不需要 GPU / 数据集）

```bash
python experiment/h3_bgfix_20260930/smoke_test.py
```

### 4.2 服务器

```bash
# 0) 先在 AutoDL 控制台把机器开机（server2 或 server3 均可，环境基本一致）
python experiment/h3_bgfix_20260930/remote.py --server server2 check   # 环境与依赖哈希核对
python experiment/h3_bgfix_20260930/remote.py --server server2 sync    # 上传本实验目录
```

四臂短预算归因（推荐）：`baseline` 先单独起，其余三臂交给带槽位的调度器。
调度器本身在服务器上 `nohup` 脱离，断掉 SSH 不影响训练：

```bash
R="python experiment/h3_bgfix_20260930/remote.py --server server2"
$R run -- "bash experiment/h3_bgfix_20260930/launch_h3_bgfix.sh baseline baseline_20k \
          --max-iters 20000 --val-interval 1000 --drop-optimizer-state"
$R run -- "cd /root/hyperseg && nohup bash experiment/h3_bgfix_20260930/screen_parallel_h3_bgfix.sh \
          20000 1000 2 loss aug full > runs/mask2former_uav/h3_bgfix_screen.nohup.log 2>&1 & echo pid=\$!"

# 进度落到服务器上（推荐：关掉笔记本也不丢历史）
$R run -- "cd /root/hyperseg && nohup bash experiment/h3_bgfix_20260930/watch_remote.sh \
          20k 300 baseline loss aug full >/dev/null 2>&1 &"

# 或者本地守候（跑完自动汇总，但关机就断）
python experiment/h3_bgfix_20260930/watch_screen.py --interval 300
```

事后一次性回收全部日志与指标（一条命令，缺失文件只报告不中断）：

```bash
$R collect --suffix 20k --local experiment/h3_bgfix_20260930/logs/collected_20k
python experiment/h3_bgfix_20260930/eta_screen.py \
    --log experiment/h3_bgfix_20260930/logs/collected_20k/h3_bgfix_watch.log
```

`watch_remote.sh` 落盘的**行格式与本地 `watch_screen.py` 完全一致**，
所以下载回来的那份可以直接喂给 `eta_screen.py --log` 和同一套解析器。
它在每一臂都写出 `exit_code` 后自行退出，不会一直挂着。

**调度器运行期间不要用 `sync`** —— bash 边读边执行脚本，覆盖
`screen_parallel_h3_bgfix.sh` 会让已加载的实例读错位置。传单个文件用 `put`。

选定后再跑全预算（**不要**加 `--drop-optimizer-state`，全预算需要能续跑）：

```bash
$R run -- "bash experiment/h3_bgfix_20260930/launch_h3_bgfix.sh full full_160k"
```

启动器会先断言 CUDA、核对骨干权重、**跑一遍冒烟测试**，通过后才 `nohup` 起训练，
日志写 `runs/mask2former_uav/h3_bgfix_<tag>_seed3407/train.log`。
训练的真实退出码由包装脚本写进同目录的 `exit_code`（`0` 表示成功）——
`nohup ... &` 之后启动器立即返回，只看它的退出码会永远读到 0。

#### 三个必须知道的坑

1. **检查点很大，先看盘**。Swin-L 检查点 **2.37 GB**（模型 0.738 GiB / 198.1M 参数，
   优化器状态 1.468 GiB）。`--drop-optimizer-state` 把它压到 0.74 GB。
   短预算筛选（不打算续跑）一律加上；需要续跑的正式训练不要加。
2. **`--cwd` 这类裸路径参数会被 Git Bash 改写**。MSYS 会把 `/root/hyperseg` 变成
   `C:/Users/.../root/hyperseg`，导致远端 `cd` 失败。写在命令字符串内部的路径不受影响；
   必须传裸路径时用 `MSYS_NO_PATHCONV=1`，或直接用 `--cwd` 的默认值（在 Python 内定义，不经 shell）。
3. **别在调度器运行时同步 `screen_*.sh`**。bash 是逐行读脚本执行的，改动正在运行的
   shell 脚本的字节会让已加载的实例错位。等它跑完再 `sync`。

### 4.3 回收与评估

```bash
python experiment/h3_bgfix_20260930/remote.py pull \
  --remote /root/hyperseg/runs/mask2former_uav/h3_bgfix_full_160k/train.log \
           /root/hyperseg/runs/mask2former_uav/h3_bgfix_full_160k/val_history.jsonl \
           /root/hyperseg/runs/mask2former_uav/h3_bgfix_full_160k/config.json \
  --local experiment/h3_bgfix_20260930/logs/
```

`test_2` 回归（无真值，只看分布）：

```bash
python experiment/h3_bgfix_20260930/regression_test2.py \
  --checkpoint runs/mask2former_uav/h3_bgfix_full_160k/best.pt \
  --reference experiment/mask2former_uav/test2_20260926/swin_l_hyperseg_复赛.zip \
  --output experiment/h3_bgfix_20260930/results/regression_full.json
```

## 5. 预算与成本

与已发布 H3 基线**完全相同的预算**：512 裁剪、micro-batch 2、160000 次 optimizer update、
PolyLR power 0.9、骨干 LR ×0.1、grad clip 1.0、seed 3407、`--with-checkpointing`。
这样唯一的差异就是本实验改造的那几项。

### 5.1 实测成本（RTX 4090，server2，2026-09-30）

| 项 | 实测 |
| --- | --- |
| 训练 | **0.194 s/update**（单进程独占），峰值显存 4.01 GB |
| 原生验证 | 699 张 @1024，约 **60 s** |
| 代理验证 | 200 张 × 3 尺度，约 **21 s** |
| 短预算单臂 | 20k update ≈ 65 min 训练 + 20 min 验证 ≈ **1.5 h** |

**并发不加速。** 单进程已把 GPU 吃到 96%；三进程并发时每进程 0.623 s/update，
聚合 4.82 updates/s，对比单进程 5.15 updates/s —— 只拿到 94%。
四臂总计 80k update 的物理下限就是约 4.3 h 纯训练，并发与否墙钟时间基本相同。
并发调度器的意义是**在显存余量内安排任务**，不是提速：
每进程约 5.2 GB，四并发只剩不到 4 GB 余量、一次激活尖峰会打挂所有臂，
所以槽位限 2–3。

## 6. 判定标准（跑之前先定好）

改造算**有效**需要同时满足：

1. **代理 mIoU 上升**（细节尺度裕度变好）；
2. **背景假阳性率下降**，且**假阴性率不显著上升**（没有把背景推向另一个极端）；
3. `test_2` 回归里 `bg>0.8` 的图数量下降（基线 100 张 / `bg>0.6` 222 张），
   且分桶表中非背景类的份额不再塌到 ~0；
4. **原生 val mIoU 不下降**（或下降幅度在种子噪声内）。

只满足 1、3 而原生掉很多，说明是在牺牲域内性能换域外，需要调小 `false_bg_weight`。
只满足 4 而 1—3 不动，说明改动没生效，先查增强是否真的走进了新分支。

## 7. 已知限制

- `test_2` 无真值，本实验**不会产生任何比赛得分**；所有结论都建立在代理集与无标注分布统计上。
- 代理集只覆盖「细节尺度」一个轴，色偏/雾/新场景类型只能靠增强的域随机化去覆盖，无法验证。
- 单一种子。按 H3 文档约定，一个种子的结论只作初步判断，定论需要 3407/3408/3409 三种子。
- 权重与数据集不入仓库，只回收日志与指标。
