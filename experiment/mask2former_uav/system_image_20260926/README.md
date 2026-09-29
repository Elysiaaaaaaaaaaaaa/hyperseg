# server2 系统盘镜像准备

本目录记录将低空航拍比赛所需代码、环境、模型和数据复制到 server2 系统盘的操作。
原始数据盘 `/root/autodl-tmp` 保留作核对；网站保存系统盘镜像后，运行路径均在 `/root` 下。

| 内容 | 系统盘路径 |
| --- | --- |
| 项目代码及 H3 Python 虚拟环境 | `/root/hyperseg`、`/root/hyperseg/.venv-mask2former` |
| MMSegmentation 源码 | `/root/mmsegmentation` |
| 带标注比赛数据 | `/root/hyperseg/dataset/low_altitude_2026/train/{images,masks}` |
| 初赛无标注图像 | `/root/hyperseg/dataset/low_altitude_2026/images` |
| 复赛无标注图像 | `/root/hyperseg/dataset/low_altitude_2026/test_2/images` |
| 固定训练、验证、有标注测试划分 | `/root/hyperseg/runs/splits/{train,val,test}.txt` |
| H3 最佳检查点 | `/root/hyperseg/runs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32/best.pt` |

迁移脚本 [`migrate_server2.sh`](migrate_server2.sh) 显式复制上述数据。
不会复制 `/root/autodl-tmp/data/test_2.zip`、数据集内重复的 `test_1.zip`、
LoveDA/SUIM 等实验结果和原项目 `runs/` 中的历史输出。项目 `models/` 和
`.venv-mask2former/` 会复制；虚拟环境引用的 `/root/miniconda3` 本就在系统盘。
`test2_20260926/prepare_data.py` 仅用于以后从原始 ZIP 重建数据；本镜像已带
校验过的解压图像，未附带重复 ZIP。

[`verify_server2.py`](verify_server2.py) 逐文件校验四个图像/掩码目录，
核对固定划分、H3 权重 SHA-256、MMSegmentation 源码及 Python 依赖，
并执行复赛推理入口的 CPU 预检。输出在
`/root/hyperseg/runs/system_image_20260926/{verification.log,verification.json}`。
只有 `verification.json` 中的 `status=passed` 才表示迁移完成。

在系统镜像中，CPU 预检命令为：

```bash
cd /root/hyperseg
PYTHONPATH=/root/mmsegmentation:/root/hyperseg \
  .venv-mask2former/bin/python \
  experiment/mask2former_uav/test2_20260926/run_inference.py --check-only
```

分配 GPU 后，以下脚本在后台对 1300 张复赛图像运行 H3 推理、检查输出格式并生成预测 ZIP：

```bash
bash /root/hyperseg/experiment/mask2former_uav/test2_20260926/launch_server2.sh
```

输出保存在 `/root/hyperseg/runs/mask2former_uav/h3_test2_20260926/`。
初赛 500 张图像可使用 `infer_h3_hyperseg.py`，显式传入
`--input /root/hyperseg/dataset/low_altitude_2026/images` 和新检查点路径。
有标注 `test.txt` 仅用于离线评估，不应与两个无标注测试目录混淆。
