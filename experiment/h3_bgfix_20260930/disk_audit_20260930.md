# server2 磁盘盘点与清理记录（2026-09-30 19:50）

## 触发

系统盘 `/` 已用 92%（30 G 中剩 2.6 G），五臂任务写检查点后有写满风险。
用户指示：系统盘的训练数据可删，数据盘还有一份。

## 一、先验证「数据盘那一份」能不能顶

删除前对每一份做内容级核对，不只比目录名。

| 系统盘 | 体积 | 数据盘对应物 | 校验方式 | 结论 |
| --- | --- | --- | --- | --- |
| `dataset/low_altitude_2026` | 15.0 G | `/root/autodl-tmp/data/low_altitude_2026` | 文件数 500 / 1300 / 6996 / 6996 全等；`train/masks` 全量聚合 sha256 两侧同为 `68b95c1d62a5a99df0bdb0440cebcf73`；按 `NR%350` 抽样 20 张 train + 10 张 test_2 逐张比对 | **逐字节一致**，且数据盘多一个 `test_1.zip`（811 MB）→ 超集 |
| `models/` | 1.9 G | `/root/autodl-tmp/hyperseg/models` | 20 个文件逐个 `stat` 尺寸比对（含 Swin-L 骨干 794 517 673 B、MiT-B3、`hyperseg_b3_best.pt`），仅目录元数据差 3 940 B | **内容一致** |
| `runs/…/h3_swin_l_hyperseg_160k_seed3407_fp32` | 2.3 G | `/root/autodl-tmp/work_dirs/…/同名` | 两份 `best.pt` 同为 2 369 198 126 B、`config.json` 同为 1 200 B | **超集**（数据盘多 `last.pt`、`train.log`、`test_eval_best/`、`unlabeled_test_predictions/`） |
| `runs/…/h3_bgfix_{baseline,loss,aug,full,loss01}_20k_seed3407` | 7.4 G | **无** | `du -sh` 逐项列出后确认数据盘 `work_dirs/` 下无同名目录 | **无副本 —— 本轮实验唯一产物** |

附带发现：已交付的 h3_swin_l 那份 `config.json` 里 `data_root`、`backbone_checkpoint`、
`work_dir` 全部指向 `/root/autodl-tmp/...`，说明**那次 160k 训练本来就是读数据盘的**，
系统盘这份是 9-26 迁移时拷过来的，对它而言纯属冗余。

## 二、执行（只删 dataset，其余原样）

用户决定：删 `dataset`，`models/` 与 `h3_swin_l` 不动；原位置留软链。

脚本 `drop_dataset_sys.sh`，三重前置断言，任一不通过即中止且不删任何东西：

1. 路径白名单（`SRC` 必须严格等于 `/root/hyperseg/dataset`，且不是已存在的软链）
2. 无 `train_h3_bgfix` / `eval_proxy` / `infer_h3_hyperseg` / `run_inference` 进程占用
3. 数据盘副本文件数齐全（6996 / 6996 / 1300 / 500）**且** `train/masks` 全量聚合哈希与系统盘一致

执行结果：

```
删除前  overlay  30G  28G  2.6G  92% /
删除后  overlay  30G  13G   18G  43% /
/root/hyperseg/dataset -> /root/autodl-tmp/data
```

走软链回读校验：`train/images` 6996、`train/masks` 6996、`test_2/images` 1300、
`images` 500，`finish_loss01_eval.sh` 依赖的 `test_2/images/test2_1.png` 可读。
未动的部分：`models/` 1.9 G、`h3_swin_l` 2.3 G、五臂 7.4 G。

## 三、同时纠正的一个误判

起初据 `df -h` 判断「数据盘还有 363 GB 空闲」，**这是错的**。

`df -h` 不带参数时看不到 `/root/autodl-tmp`，只列出 `/autodl-pub` 那行
`7.0T / 6.7T / 363G avail`——那 376 G 属于**只读公共数据区**。
可写数据盘挂的是同一块 `/dev/md0`，但带 `prjquota`：

```
/dev/md0 on /root/autodl-tmp type xfs (rw,...,prjquota)
$ df -h /root/autodl-tmp
/dev/md0  50G  48G  2.8G  95%  /root/autodl-tmp
```

**本实例数据盘配额 50 G，已用 48 G，只剩 2.8 G。**

这也说明「把五臂移到数据盘」当时根本行不通 —— 7.4 G 放不下。
五臂留在系统盘（现余 18 G）是当下唯一可行且安全的安排。

## 四、数据盘可回收项（未执行）

数据盘 95% 一旦写满，`finish_loss01_eval.sh` 与后续训练都会受影响。候选：

| 路径 | 体积 | 说明 |
| --- | --- | --- |
| `work_dirs/mask2former_uav/h3_smoke_pretrained/` | 4.5 G | 9-18 冒烟产物，`best.pt` + `last.pt` 各 2.37 G，纯调试残留 |
| `hyperseg_manual_v2_20260912/runs/` | 4.9 G | 9-12 旧手工实验产物 |
| `data/test_2.zip` + `data/low_altitude_2026/test_1.zip` | 2.8 G | 压缩包，内容已解压在同树（`test_2.zip` 内 1301 项 ↔ 解压目录 1300 png） |
| `hyperseg_ablation_v2_20260912/runs/` | 1.1 G | 9-12 消融实验产物 |
| `work_dirs/mask2former_uav/h1_smoke_retry1/` | 245 M | 冒烟重试残留 |

合计约 **13 G**，可把数据盘压回 ~30%。

**不可动**：`data/low_altitude_2026/`（现已是系统盘软链的目标，唯一数据集）、
`hyperseg/models/`、`work_dirs/…/h3_swin_l_hyperseg_160k_seed3407_fp32/`（系统盘那份的备份）。
