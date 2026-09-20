import torch
import torch.nn.functional as F


def hyperseg_loss(
    output,
    target,
    class_weights=None,
    focal_gamma=1.5,
    boundary_weight=0.1,
    rare_weight=0.5,
    ignore_index=0,
):
    logits = output["logits"]
    ce = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )
    probability = logits.softmax(1)
    valid = target != ignore_index
    safe_target = target.clamp(0, logits.shape[1] - 1)
    focal = (1 - probability.gather(1, safe_target[:, None]).squeeze(1)).pow(focal_gamma)
    ce = (ce[valid] * focal[valid]).mean() if valid.any() else logits.new_zeros(())
    dice = []
    for cls in range(1, logits.shape[1]):
        actual = (target == cls) & valid
        if actual.any():
            pred = probability[:, cls][valid]
            truth = actual[valid].float()
            dice.append(1 - (2 * (pred * truth).sum() + 1) / (pred.sum() + truth.sum() + 1))
    dice = torch.stack(dice).mean() if dice else logits.new_zeros(())
    rare = logits.new_zeros(())
    for cls in (5, 7, 8):
        actual = (target == cls) & valid
        if actual.any():
            rare = rare + (-probability[:, cls][actual].clamp_min(1e-6).log()).mean()
    boundary_target = torch.zeros_like(target, dtype=torch.bool)
    boundary_target[:, 1:] |= target[:, 1:] != target[:, :-1]
    boundary_target[:, :, 1:] |= target[:, :, 1:] != target[:, :, :-1]
    boundary_target = boundary_target.float()
    bce = F.binary_cross_entropy_with_logits(output["boundary"].squeeze(1), boundary_target, reduction="none")
    total = 0.55 * ce + 0.3 * dice + rare_weight * 0.05 * rare
    total = total + boundary_weight * bce[valid].mean()
    return torch.nan_to_num(total, nan=0.0)


@torch.no_grad()
def mean_iou(logits, target, classes=9, ignore_index=0):
    prediction = logits.argmax(1)
    scores = []
    for cls in range(1, classes):
        valid = target != ignore_index
        union = ((prediction == cls) | (target == cls)) & valid
        if union.any():
            scores.append(float((((prediction == cls) & (target == cls)) & valid).sum() / union.sum()))
    return sum(scores) / max(1, len(scores))
