"""
train_fewshot_v2.py

Optimized training entry point for Mondstadt UAV Few-shot + HyperNetwork.

Important:
- Uses fewshot_dataset_v2.py.
- True train/val split.
- query_size=2 by default to increase useful GPU work.
- batch_size=2 by default.
- prefetch_factor and persistent workers enabled.
- Removes per-step CPU mIoU calculation from the hot path.
- Prints data/forward/backward timing so GPU starvation is obvious.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Make the repository root importable when this file is launched directly
# (``python tools/train_fewshot.py``).  Python otherwise only adds the
# ``tools`` directory to ``sys.path``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
try:
    # PyTorch 2.x API.  The legacy torch.cuda.amp.autocast does not accept
    # ``device_type`` on some installed versions.
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover - compatibility with older PyTorch
    from torch.cuda.amp import GradScaler, autocast

_NEW_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")

def amp_autocast(device_type, enabled):
    if _NEW_AMP:
        return autocast(device_type=device_type, enabled=enabled)
    return autocast(enabled=enabled)
from torch.utils.data import DataLoader

from fewshot_hyperseg.model import HyperSegUAV, NUM_CLASSES, IGNORE_INDEX
from fewshot_hyperseg.losses import HyperSegLoss, mean_iou

DEFAULT_ROOT = "/home/apocalypse/code/Mondstadt/dataset"
DEFAULT_CLASSES = [2, 3, 4, 5, 6, 7, 8]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def collate_episodes(batch):
    return {
        "support_images": torch.stack([x["support_images"] for x in batch]),
        "support_masks": torch.stack([x["support_masks"] for x in batch]),
        "query_images": torch.stack([x["query_images"] for x in batch]),
        "query_masks": torch.stack([x["query_masks"] for x in batch]),
        "class_ids": batch[0]["class_ids"],
        "episode_id": [x["episode_id"] for x in batch],
    }


def build_dataset(args, split):
    from fewshot_hyperseg.fewshot_dataset import UAVFewShotDataset
    return UAVFewShotDataset(
        data_root=args.dataset_root,
        n_way=args.n_way,
        k_shot=args.k_shot,
        query_size=args.query_size,
        image_size=args.image_size,
        target_image_size=args.loss_image_size,
        episodes_per_epoch=(
            args.train_episodes if split == "train"
            else args.val_episodes
        ),
        class_ids=args.class_ids,
        split=split,
        train=(split == "train"),
        seed=args.seed,
        min_pixels=args.min_pixels,
        val_ratio=args.val_ratio,
    )


def build_loader(dataset, args, train):
    kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size if train else args.val_batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_episodes,
    )
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs)


def build_model(args, device):
    return HyperSegUAV(
        num_classes=NUM_CLASSES,
        class_ids=args.class_ids,
        k_shot=args.k_shot,
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        prototype_dim=args.prototype_dim,
        context_dim=args.context_dim,
        fusion_dim=args.fusion_dim,
        adapter_rank=args.adapter_rank,
        freeze_backbone=args.freeze_backbone,
    ).to(device)


def build_optimizer(model, args):
    backbone, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if n.startswith("backbone.") else head).append(p)

    groups = []
    if backbone:
        groups.append({
            "params": backbone,
            "lr": args.lr_backbone,
            "weight_decay": args.weight_decay,
        })
    if head:
        groups.append({
            "params": head,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        })
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def move_batch(batch, device):
    si = batch["support_images"].to(device, non_blocking=True)
    sm = batch["support_masks"].to(device, non_blocking=True)
    qi = batch["query_images"].to(device, non_blocking=True)
    qm = batch["query_masks"].to(device, non_blocking=True)
    return si, sm, qi, qm


def flatten_query_target(logits, target):
    # Model output is [B,Q,C,H,W]; losses are written for [B,C,H,W].
    if logits.ndim == 5:
        B, Q, C, H, W = logits.shape
        logits = logits.reshape(B * Q, C, H, W)
    if target.ndim == 4:
        target = target.reshape(-1, target.shape[-2], target.shape[-1])
    return logits, target


def run_epoch(model, loader, criterion, optimizer, scaler, device,
              train=True, amp=True, grad_clip=1.0, log_interval=20):
    model.train(train)
    total_loss = 0.0
    total_steps = 0
    data_time_sum = forward_time_sum = backward_time_sum = 0.0

    end = time.perf_counter()
    for step, batch in enumerate(loader):
        data_time = time.perf_counter() - end
        data_time_sum += data_time

        si, sm, qi, qm = move_batch(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.set_grad_enabled(train):
            with amp_autocast(device.type, amp and device.type == "cuda"):
                outputs = model(
                    query_images=qi,
                    support_images=si,
                    support_masks=sm,
                )
                logits, target = flatten_query_target(
                    outputs["logits"], qm
                )
                if logits.shape[-2:] != target.shape[-2:]:
                    logits = torch.nn.functional.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)

                loss_outputs = dict(outputs)
                loss_outputs["logits"] = logits
                if loss_outputs.get("boundary") is not None:
                    b = loss_outputs["boundary"]
                    if b.ndim == 5:
                        loss_outputs["boundary"] = b.reshape(
                            -1, b.shape[2], b.shape[3], b.shape[4]
                        )
                    b = loss_outputs["boundary"]
                    if b.shape[-2:] != target.shape[-2:]:
                        loss_outputs["boundary"] = torch.nn.functional.interpolate(
                            b, size=target.shape[-2:], mode="bilinear", align_corners=False
                        )

                loss, details = criterion(loss_outputs, target)

        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - t0
        forward_time_sum += forward_time

        backward_time = 0.0
        if train:
            if device.type == "cuda":
                torch.cuda.synchronize()
            tb = time.perf_counter()

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )
            scaler.step(optimizer)
            scaler.update()

            if device.type == "cuda":
                torch.cuda.synchronize()
            backward_time = time.perf_counter() - tb
            backward_time_sum += backward_time

        total_loss += float(loss.detach())
        total_steps += 1

        if (step + 1) % log_interval == 0:
            avg_data = data_time_sum / total_steps
            avg_fwd = forward_time_sum / total_steps
            avg_bwd = backward_time_sum / max(total_steps, 1)
            print(
                f"[{'train' if train else 'val'}] "
                f"{step+1}/{len(loader)} "
                f"loss={total_loss/total_steps:.4f} "
                f"data={avg_data:.3f}s "
                f"fwd={avg_fwd:.3f}s "
                f"bwd={avg_bwd:.3f}s"
            )

        end = time.perf_counter()

    return {
        "loss": total_loss / max(total_steps, 1),
        "data_time": data_time_sum / max(total_steps, 1),
        "forward_time": forward_time_sum / max(total_steps, 1),
        "backward_time": backward_time_sum / max(total_steps, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp=True):
    model.eval()
    loss_sum = 0.0
    miou_sum = 0.0
    n = 0

    for batch in loader:
        si, sm, qi, qm = move_batch(batch, device)
        with amp_autocast(device.type, amp and device.type == "cuda"):
            outputs = model(
                query_images=qi,
                support_images=si,
                support_masks=sm,
            )
            logits, target = flatten_query_target(outputs["logits"], qm)
            if logits.shape[-2:] != target.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
            loss_outputs = dict(outputs)
            loss_outputs["logits"] = logits
            if loss_outputs.get("boundary") is not None:
                b = loss_outputs["boundary"]
                if b.ndim == 5:
                    loss_outputs["boundary"] = b.reshape(
                        -1, b.shape[2], b.shape[3], b.shape[4]
                    )
                b = loss_outputs["boundary"]
                if b.shape[-2:] != target.shape[-2:]:
                    loss_outputs["boundary"] = torch.nn.functional.interpolate(
                        b, size=target.shape[-2:], mode="bilinear", align_corners=False
                    )
            loss, _ = criterion(loss_outputs, target)

        # Validation mIoU is intentionally calculated only here.
        miou, _ = mean_iou(logits.float(), target)
        loss_sum += float(loss)
        miou_sum += float(miou)
        n += 1

    return {
        "loss": loss_sum / max(n, 1),
        "miou": miou_sum / max(n, 1),
    }


def save_ckpt(path, model, optimizer, scheduler, scaler, epoch, best, args):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "best_miou": best,
        "args": vars(args),
    }, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=DEFAULT_ROOT)
    p.add_argument("--n-way", type=int, default=7)
    p.add_argument("--k-shot", type=int, default=1, choices=[1, 5, 10, 20])
    p.add_argument("--class-ids", type=int, nargs="+", default=DEFAULT_CLASSES)
    p.add_argument("--query-size", type=int, default=2)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--loss-image-size", type=int, default=1024)
    p.add_argument("--train-episodes", type=int, default=1000)
    p.add_argument("--val-episodes", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--val-batch-size", type=int, default=1)
    p.add_argument("--min-pixels", type=int, default=100)
    p.add_argument("--val-ratio", type=float, default=0.1)

    p.add_argument("--backbone", default="nvidia/mit-b3")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--prototype-dim", type=int, default=256)
    p.add_argument("--context-dim", type=int, default=256)
    p.add_argument("--fusion-dim", type=int, default=128)
    p.add_argument("--adapter-rank", type=int, default=32)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--lambda-boundary", type=float, default=0.10)
    p.add_argument("--lambda-proto", type=float, default=0.05)
    p.add_argument("--lambda-fusion", type=float, default=0.01)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--output-dir", default="./checkpoints/fewshot_v2")
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args()


def main():
    """组织一次完整的 episodic few-shot 训练。

    这里的 ``main`` 只负责串联训练框架；具体职责分散在几个辅助函数中：

    - ``build_dataset``：生成由 support/query 组成的 episode；
    - ``build_model``：创建由 support 原型条件化的 HyperSegUAV；
    - ``run_epoch``：在 query mask 的监督下执行反向传播；
    - ``evaluate``：固定参数，在验证 episode 上计算 loss 和 mIoU。

    注意：这不是在每个 episode 内临时微调模型。support 图像和 mask 会在
    forward 中生成原型及 HyperNetwork 参数，query loss 则直接更新整个模型。
    """
    # 1. 读取命令行超参数。args 同时控制 episode 的 N/K/Q、模型规模、
    #    优化策略和 checkpoint 路径，稍后也会完整写入 args.json 方便复现。
    args = parse_args()
    if args.resume:
        # 断点里已经包含 backbone 权重，因此恢复训练时只创建网络结构，
        # 不再请求 Hugging Face 的预训练权重，适合离线训练服务器。
        args.pretrained = False

    # 固定 Python、NumPy 和 PyTorch 的随机种子。cuDNN 仍启用了 benchmark，
    # 所以该设置主要保证数据划分和 episode 采样可复现，并非严格逐位复现。
    seed_everything(args.seed)

    # 2. 准备 GPU 运行环境。这份优化版入口将 CUDA 作为硬性要求，不提供
    #    CPU fallback；TF32 可加速支持它的 NVIDIA GPU 上的矩阵乘和卷积。
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this optimized training script.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # PyTorch 2.x can improve transformer/conv throughput.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # 3. 创建实验目录，并在真正训练前保存所有命令行参数。即使训练中断，
    #    也能通过 args.json 确认该次实验使用的 N-way、K-shot、尺寸和学习率。
    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"GPU count: {torch.cuda.device_count()}")

    # 4. 分别建立训练和验证的 episodic 数据集。
    #
    #    每次取样返回一个 episode：
    #      support_images/support_masks: N*K 个带标签的支持样本；
    #      query_images/query_masks:     Q 个需要预测并计算 loss 的查询样本。
    #
    #    两个数据集会基于相同 seed 建立互斥的 train/val 图像划分，防止
    #    同一原图同时出现在训练 episode 和验证 episode 中。
    train_ds = build_dataset(args, "train")
    val_ds = build_dataset(args, "val")

    # DataLoader 的 batch 维表示“一批 episode”，并不是普通的一批单图。
    # 例如默认 B=2、N=7、K=1、Q=2 时，一个 batch 含 14 张 support 和
    # 4 张 query；collate_episodes 会把它们整理成 [B,S,...]/[B,Q,...]。
    train_loader = build_loader(train_ds, args, True)
    val_loader = build_loader(val_ds, args, False)

    # 5. 创建模型和组合损失。
    #
    #    模型使用 support mask 从 support 特征中提取逐类原型，再由
    #    HyperNetwork 产生动态调制、融合和门控参数，最后预测 query。
    #    criterion 只拿 query mask 做主要监督，同时加入边界、原型分离和
    #    融合权重正则项；标签 0 始终作为 Ignore 排除。
    model = build_model(args, device)
    criterion = HyperSegLoss(
        ignore_index=IGNORE_INDEX,
        lambda_boundary=args.lambda_boundary,
        lambda_proto=args.lambda_proto,
        lambda_fusion=args.lambda_fusion,
    ).to(device)

    # 6. 配置训练状态。
    #
    #    build_optimizer 将 backbone 和其余模块分组：backbone 默认使用较小的
    #    lr_backbone，其余 HyperNetwork/decoder/head 使用 lr。余弦调度器按
    #    epoch 将学习率逐渐降至 min_lr；GradScaler 则负责 CUDA AMP 的缩放。
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    try:
        scaler = GradScaler("cuda", enabled=args.amp)
    except TypeError:  # legacy torch.cuda.amp.GradScaler
        scaler = GradScaler(enabled=args.amp)

    best_miou = -1.0
    start_epoch = 0
    if args.resume:
        # 7. 完整恢复训练现场，而不只是恢复模型参数。optimizer、scheduler
        #    和 AMP scaler 一并恢复后，学习率曲线及数值缩放可以接着上次跑。
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", -1)) + 1
        best_miou = float(state.get("best_miou", -1.0))
        print(f"Resumed {args.resume} at epoch {start_epoch}, best_mIoU={best_miou:.4f}")

    # 8. 正式训练前做一次无梯度 warm-up。
    #
    #    这一步会真实读取一个 episode 并跑完整 forward，用于尽早暴露
    #    support/query 维度、显存或模型接口错误，但不会反向传播或更新权重。
    #    打印出的 shape 也方便确认当前 N*K、Q 和输出类别数是否符合预期。
    print("\n=== Warm-up ===")
    warm = next(iter(train_loader))
    si, sm, qi, qm = move_batch(warm, device)
    with torch.no_grad():
        with amp_autocast("cuda", args.amp):
            o = model(qi, si, sm)
    print("support:", tuple(si.shape))
    print("query  :", tuple(qi.shape))
    print("logits :", tuple(o["logits"].shape))
    print("GPU mem:", f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    del warm, si, sm, qi, qm, o
    torch.cuda.empty_cache()

    # 9. epoch 级训练主循环。
    #
    #    run_epoch 内部对每批 episode 执行：
    #      support/query 前向 -> query loss -> AMP 反传 -> 梯度裁剪 -> 更新；
    #    evaluate 则只做前向，在验证 episode 上统计 loss 和 mIoU。
    print("\n=== Training ===")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_m = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            train=True, amp=args.amp, grad_clip=args.grad_clip,
            log_interval=args.log_interval
        )
        val_m = evaluate(
            model, val_loader, criterion, device, amp=args.amp
        )

        # 调度器每个 epoch 更新一次；因此下面保存的 checkpoint 包含的是
        # 已经 step 后、可供下一 epoch 继续使用的 scheduler 状态。
        scheduler.step()

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_m['loss']:.4f} "
            f"val_loss={val_m['loss']:.4f} "
            f"val_mIoU={val_m['miou']:.4f}"
        )
        print(
            f"Timing: data={train_m['data_time']:.3f}s "
            f"forward={train_m['forward_time']:.3f}s "
            f"backward={train_m['backward_time']:.3f}s"
        )

        # 10. 每轮覆盖 last.pt，保存最新模型以及继续训练所需的全部状态。
        #     best.pt 只在验证 mIoU 创历史新高时覆盖，用作最终模型选择。
        last = Path(args.output_dir) / "last.pt"
        save_ckpt(last, model, optimizer, scheduler, scaler,
                   epoch, best_miou, args)

        if val_m["miou"] > best_miou:
            best_miou = val_m["miou"]
            save_ckpt(
                Path(args.output_dir) / "best.pt",
                model, optimizer, scheduler, scaler,
                epoch, best_miou, args
            )
            print(f"Saved best.pt, mIoU={best_miou:.4f}")


if __name__ == "__main__":
    main()
