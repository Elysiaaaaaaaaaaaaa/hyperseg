"""Train M0--M4 or evaluate a checkpoint at native image resolution.

这个脚本是单次 MathSeg-UAV 实验的实际执行入口。它既可以：

* 检查数据和固定划分是否完整；
* 从头训练一个 M0--M4 变体；
* 从 ``last.pt`` 恢复训练；
* 加载 ``best.pt``/``last.pt``，在验证集或测试集上评估。

多变体、多 seed 的队列调度由同目录的 ``run.py`` 负责；本文件负责其中
每一个具体实验的完整生命周期。
"""

# 标准库：命令行参数、JSON 配置/日志、运行环境信息、计时和路径处理。
import argparse
import json
import platform
import sys
import time
from pathlib import Path

# 项目根目录。train.py 位于 ``experiment/mathseg_uav`` 下，因此向上两级。
ROOT = Path(__file__).resolve().parents[2]
# 允许用户直接执行 ``python experiment/mathseg_uav/train.py`` 时导入项目包。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 深度学习框架和模型配置库。
import torch
import transformers

# MathSeg 模型、变体名称，以及数据/损失/评估/checkpoint 辅助函数。
from experiment.mathseg_uav.model import MathSegUAV, PRIOR_NAMES, VARIANTS
from experiment.mathseg_uav.protocol import (
    class_weights, evaluate, make_loader, read_splits, resolve_dirs, restore_rng,
    rng_state, scan_masks, seed_everything, segmentation_loss, write_json,
)


def parse_args():
    """解析命令行参数，并在启动训练前做基础合法性检查。"""
    parser = argparse.ArgumentParser(description=__doc__)

    # 实验和数据相关参数。
    # ``variant`` 在恢复/评估 checkpoint 时可以省略，届时从 checkpoint 配置中读取。
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "runs/splits")
    parser.add_argument("--work-dir", type=Path)

    # 骨干网络和 MathSeg 数学先验分支的配置。
    # 默认只使用本地 Hugging Face 文件；--allow-download 才允许联网下载。
    parser.add_argument("--model-name", default="nvidia/mit-b3")
    parser.add_argument("--allow-download", action="store_true", help="allow Hugging Face network loading")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--prior-channels", nargs="+", choices=PRIOR_NAMES, default=list(PRIOR_NAMES))
    parser.add_argument("--prior-mode", choices=("normal", "zero", "shuffle"), default="normal")

    # 训练循环参数。
    # max-updates 是优化器更新次数，不是 DataLoader epoch 数。
    # accumulation 用于梯度累积，有效 batch size = batch-size * accumulation。
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=160000)
    parser.add_argument("--val-interval", type=int, default=2800)
    parser.add_argument("--val-size", type=int, default=0,
                        help="validation resize for smoke tests; 0 keeps native resolution")
    parser.add_argument("--save-interval", type=int, default=2800)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--stop-after", type=int, help="stop/save at this absolute update; retain full LR horizon")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gamma-max", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu (explicit smoke testing)")
    parser.add_argument("--amp", action="store_true", help="CUDA float16; FP32 is the default")

    # 三种互斥的特殊模式：恢复训练、评估 checkpoint、只检查数据。
    # 不指定它们时才进入普通的从头训练流程。
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path)
    group.add_argument("--eval-checkpoint", type=Path)
    group.add_argument("--check-only", action="store_true", help="scan labels/alignment in all three splits")
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    args = parser.parse_args()

    # 训练循环不能接受零或负数的关键参数；val-size=0 有特殊含义，表示保留原始分辨率。
    positive = (args.width, args.size, args.batch_size, args.accumulation, args.max_updates,
                args.val_interval, args.save_interval, args.log_interval, args.lr, args.backbone_lr_multiplier)
    if min(positive) <= 0 or args.val_size < 0 or args.num_workers < 0 or args.gamma_max < 0 or args.weight_decay < 0:
        parser.error("Invalid nonpositive training parameter")

    # stop-after 用于短程 smoke test 或主动暂停，但不能超过正式训练总预算。
    if args.stop_after is not None and not 0 < args.stop_after <= args.max_updates:
        parser.error("--stop-after must be in 1..max-updates")

    # 同一个数学先验通道重复传入会导致实验协议含义不明确，直接拒绝。
    if len(set(args.prior_channels)) != len(args.prior_channels):
        parser.error("--prior-channels contains duplicates")
    return args


def contract(args, hashes):
    """生成实验协议指纹，用于保证断点恢复时训练设置没有被悄悄改变。

    数据路径和 worker 数不放入 contract，因此同一 checkpoint 可以迁移到
    另一台机器；split 文件的 hash 会保留，避免训练/验证样本划分发生变化。
    """
    # Paths and worker counts may change when moving a run to another machine.
    keys = ("variant", "width", "prior_channels", "prior_mode", "size", "batch_size", "accumulation",
            "max_updates", "val_size", "lr", "backbone_lr_multiplier", "weight_decay", "gamma_max", "seed", "amp")
    return {"version": 1, "splits": hashes, **{key: getattr(args, key) for key in keys}}


def save_checkpoint(path, model, optimizer, scaler, update, best, metrics, args, hashes, counts):
    """保存可恢复的完整训练状态，而不只是模型权重。

    先写入同目录临时文件，再原子替换目标文件，降低训练进程中断时留下
    损坏 checkpoint 的风险。checkpoint 同时保存优化器、AMP scaler、随机数
    状态、当前 update、实验协议和 M4 类别统计。
    """
    temporary = path.with_suffix(".tmp")
    torch.save({
        "experiment": "mathseg_uav_v1", "model": model.state_dict(), "model_config": model.model_config,
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "update": update,
        "best_mIoU": best, "metrics": metrics, "rng": rng_state(),
        "contract": contract(args, hashes), "class_counts": counts,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }, temporary)
    temporary.replace(path)


def load_state(path):
    """读取并校验 MathSeg checkpoint，拒绝误加载 HyperSeg checkpoint。"""
    # Only load trusted experiment files: optimizer/RNG state requires pickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("experiment") != "mathseg_uav_v1":
        raise ValueError("Expected a MathSeg checkpoint; HyperSeg checkpoints are incompatible")
    return state


def main():
    """执行一次数据检查、评估、从头训练或断点恢复任务。"""

    # ------------------------------------------------------------------
    # 1. 解析数据目录和固定划分
    # ------------------------------------------------------------------
    args = parse_args()
    images, masks = resolve_dirs(args.data_root, args.image_dir, args.mask_dir)
    splits, hashes = read_splits(args.split_dir, images, masks)

    # --check-only 不创建模型、不占用 GPU，只扫描 train/val/test 三个划分，
    # 检查图像/掩码是否配对、尺寸是否对齐、标签是否落在 0..8。
    if args.check_only:
        counts = {name: scan_masks(images, masks, ids) for name, ids in splits.items()}
        print(json.dumps({"samples": {k: len(v) for k, v in splits.items()}, "split_hashes": hashes,
                          "class_counts": counts}, indent=2))
        return

    # ------------------------------------------------------------------
    # 2. 设备、checkpoint 和实验目录准备
    # ------------------------------------------------------------------
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu only for small smoke tests")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA")

    # 恢复训练和评估都需要先读取 checkpoint；普通从头训练时 state=None。
    state = load_state(args.resume or args.eval_checkpoint) if args.resume or args.eval_checkpoint else None

    # 未显式指定 variant 时，从 checkpoint 的 model_config 恢复；全新实验默认 M0。
    args.variant = args.variant or (state["model_config"]["variant"] if state else "M0")
    args.work_dir = args.work_dir or (args.resume.parent if args.resume else
                                    ROOT / "runs/mathseg_uav" / f"{args.variant}_seed{args.seed}")

    # checkpoint 的模型变体和本次命令必须一致；恢复训练还必须完全匹配训练协议。
    if state and args.variant != state["model_config"]["variant"]:
        raise ValueError("Requested variant differs from checkpoint")
    if args.resume and state["contract"] != contract(args, hashes):
        differences = [k for k, v in contract(args, hashes).items() if state["contract"].get(k) != v]
        raise ValueError(f"Resume protocol differs: {differences}; reuse original training arguments")
    if args.eval_checkpoint and hashes != state["contract"]["splits"]:
        raise ValueError("Evaluation split hashes differ from the training protocol")

    # 评估模式不写新的训练目录；从头训练/恢复训练则要求目录存在且不能覆盖已有实验。
    if not args.eval_checkpoint:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        if not args.resume and any((args.work_dir / name).exists() for name in ("config.json", "last.pt", "best.pt")):
            raise FileExistsError("Run already exists; use --resume or a new --work-dir")

    # 固定 Python/PyTorch/数据加载相关随机性，使实验尽可能可复现。
    seed_everything(args.seed)

    # ------------------------------------------------------------------
    # 3. 构造模型，并根据运行模式加载模型权重
    # ------------------------------------------------------------------
    # 从 checkpoint 恢复时使用 checkpoint 内保存的完整 model_config，避免当前
    # 命令行默认值变化导致模型结构不一致；从头训练时使用命令行参数创建模型。
    model = (MathSegUAV(**state["model_config"]) if state else MathSegUAV(
        variant=args.variant, width=args.width, model_name=args.model_name,
        pretrained=not args.no_pretrained, local_files_only=not args.allow_download,
        prior_channels=args.prior_channels, prior_mode=args.prior_mode,
    )).to(device)
    if state:
        model.load_state_dict(state["model"], strict=True)

    # ------------------------------------------------------------------
    # 4. checkpoint 评估模式：评估后直接返回，不进入训练循环
    # ------------------------------------------------------------------
    if args.eval_checkpoint:
        loader = make_loader(images, masks, splits[args.eval_split], workers=args.num_workers)
        metrics = evaluate(model, loader, device)
        metrics.update(checkpoint=str(args.eval_checkpoint), split=args.eval_split,
                       variant=args.variant, split_hashes=hashes, inference="native_whole_fp32_no_tta")
        output = args.eval_checkpoint.with_name(args.eval_checkpoint.stem + f"_{args.eval_split}_metrics.json")
        write_json(output, metrics)
        print(json.dumps(metrics, indent=2))
        return

    # ------------------------------------------------------------------
    # 5. 构造优化器、AMP scaler，并准备恢复状态
    # ------------------------------------------------------------------
    # 编码器使用较小学习率，MathSeg 新增模块/解码头使用完整学习率。
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr * args.backbone_lr_multiplier},
        {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    start, best, metrics = 0, -1.0, None
    counts = None

    # M4 是类别重加权变体：只用 train 划分统计类别频率，不能使用验证/测试标签。
    # 恢复时沿用 checkpoint 内保存的统计，确保中断前后权重协议一致。
    if args.variant == "M4":
        counts = state["class_counts"] if state else scan_masks(images, masks, splits["train"])
        if not counts or sum(counts[1:]) == 0:
            raise ValueError("M4 needs valid training class counts")
        write_json(args.work_dir / "class_counts.json", {"counts": counts, "train_hash": hashes["train"]})

    # 断点恢复：恢复优化器、AMP scaler、当前 update、最佳验证分数和最近指标。
    # 模型权重已在上面的 model.load_state_dict 中恢复。
    if args.resume:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, best, metrics = state["update"], state["best_mIoU"], state["metrics"]
    stop = args.stop_after or args.max_updates
    if start >= stop:
        raise ValueError(f"Checkpoint is already at update {start}; requested stop is {stop}")

    # 训练 loader 从对应 microbatch 位置开始，避免恢复后重复消费已经训练过的数据。
    # 验证 loader 默认保留原始分辨率；--val-size 只建议用于短程 smoke test。
    loader = make_loader(images, masks, splits["train"], workers=args.num_workers, size=args.size,
                         seed=args.seed, batch_size=args.batch_size,
                         start_microbatch=start * args.accumulation,
                         total_microbatches=stop * args.accumulation)
    val_loader = make_loader(images, masks, splits["val"], workers=args.num_workers,
                             size=args.val_size or None)

    # 保存本次实验的完整配置和环境信息，便于之后审计参数、复现实验或迁移到服务器。
    configuration = {"args": vars(args), "contract": contract(args, hashes), "model_config": model.model_config,
                     "train_samples": len(splits["train"]), "val_samples": len(splits["val"]),
                     "parameters": sum(p.numel() for p in model.parameters()),
                     "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                     "transformers": transformers.__version__,
                                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                                     "cuda": torch.version.cuda,
                                     "device_requested": str(device)}}
    write_json(args.work_dir / ("resume_config.json" if args.resume else "config.json"), configuration)
    write_json(args.work_dir / "status.json", {"completed": False, "update": start,
                                               "max_updates": args.max_updates, "state": "running"})
    print(json.dumps(configuration, default=str), flush=True)

    # iterator 由训练循环持续取 batch；恢复时把随机数状态放回 checkpoint 保存的时刻。
    iterator = iter(loader)
    if args.resume:
        restore_rng(state["rng"])
    started = time.monotonic()

    # ------------------------------------------------------------------
    # 6. 主训练循环：一次 update 可能包含多个 gradient accumulation microbatch
    # ------------------------------------------------------------------
    for update in range(start + 1, stop + 1):
        model.train()

        # Poly 学习率衰减：保持 backbone/新增模块的学习率比例不变。
        factor = (1 - (update - 1) / args.max_updates) ** 0.9
        for group, multiplier in zip(optimizer.param_groups, (args.backbone_lr_multiplier, 1)):
            group["lr"] = args.lr * multiplier * factor

        # M4 的类别权重随训练进度从 0 平滑增加到 gamma_max；其他变体不加权。
        weights = class_weights(counts, update / args.max_updates, args.gamma_max).to(device) if counts else None
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0

        # 梯度累积期间每个 microbatch 的 loss 除以 accumulation，
        # 这样累积后的梯度规模与直接使用有效 batch 的结果一致。
        for _ in range(args.accumulation):
            batch = next(iterator)

            # AMP 开启时使用 float16 autocast；默认 FP32。模型输入和标签异步搬到目标设备。
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                output = model(batch["image"].to(device, non_blocking=True))
                loss = segmentation_loss(output, batch["mask"].to(device, non_blocking=True), weights) / args.accumulation

            # 非有限 loss 直接中止，避免 NaN/Inf 污染模型和 checkpoint。
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at update {update}")
            scaler.scale(loss).backward()
            loss_value += float(loss.detach())

        # AMP 下先还原梯度尺度，再做梯度裁剪；随后更新参数并更新 scaler。
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()

        # 按固定间隔写 JSON 日志，记录 loss、学习率、梯度范数和耗时。
        if update % args.log_interval == 0 or update == start + 1:
            record = {"update": update, "loss": loss_value, "lr": optimizer.param_groups[1]["lr"],
                      "grad_norm": float(norm), "seconds_per_update_including_val": (time.monotonic() - started) / (update - start)}
            if weights is not None:
                record["class_weights"] = weights.tolist()
            print(json.dumps(record), flush=True)

        # 到验证间隔或训练终点时，在 val split 上计算 mIoU 和其他指标。
        if update % args.val_interval == 0 or update == stop:
            metrics = {"update": update, **evaluate(model, val_loader, device)}
            if metrics["mIoU"] is None:
                raise ValueError("Validation set contains no valid pixels")
            print(json.dumps(metrics), flush=True)
            with (args.work_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics) + "\n")
            write_json(args.work_dir / "val_metrics.json", metrics)

            # best.pt 只保存验证 mIoU 创新高的模型；best_metrics.json 保存对应指标。
            if metrics["mIoU"] > best:
                best = metrics["mIoU"]
                save_checkpoint(args.work_dir / "best.pt", model, optimizer, scaler, update, best, metrics, args, hashes, counts)
                write_json(args.work_dir / "best_metrics.json", metrics)

        # last.pt 是可继续训练的最近断点，保存频率通常比验证更密；
        # 在验证点和训练终点也强制保存，确保重要状态不会丢失。
        if update % args.save_interval == 0 or update % args.val_interval == 0 or update == stop:
            save_checkpoint(args.work_dir / "last.pt", model, optimizer, scaler, update, best, metrics, args, hashes, counts)

    # ------------------------------------------------------------------
    # 7. 训练结束：区分完整跑完 max-updates 和 stop-after 主动暂停
    # ------------------------------------------------------------------
    write_json(args.work_dir / "status.json", {"completed": stop == args.max_updates, "update": stop,
                                               "max_updates": args.max_updates, "best_mIoU": best})
    print(f"{'completed' if stop == args.max_updates else 'paused'}: {args.work_dir}", flush=True)


if __name__ == "__main__":
    main()
