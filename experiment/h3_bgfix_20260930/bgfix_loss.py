"""Loss for the H3 background-collapse fix.

The original ``hyperseg_uav.losses.hyperseg_loss`` is::

    0.55 * focal_CE + 0.30 * mean_class_dice + 0.05 * rare_class_loss + 0.10 * boundary_BCE

with ``class_weights=None`` -- the parameter exists but no training entry ever passed it.

Why that lets the model collapse onto Background (measured on ``test_2``):

* Background is 27% of the labelled pixels, so it is present in every batch and its Dice
  term is already near saturation -> the Dice branch contributes almost no gradient for it.
* The focal CE term is largest for genuinely hard pixels regardless of class, so on an
  out-of-distribution input the cheapest way to reduce loss is to route every uncertain
  pixel into whichever class already owns the region -- Background.
* No term anywhere penalises predicting Background where the label is *not* Background.

On the labelled test split the model is fine (Background IoU 0.711, Building 0.853,
Road 0.817), so this is not a globally background-biased model: the aggregate excess of
Background on ``test_2`` comes entirely from the ~18% of images that collapse.  A blanket
down-weight of Background is therefore the wrong fix, and it would also damage the urban
parcels where large uniform pavement really *is* the Background class.

The added term is targeted instead::

    false_background = mean over {valid pixels with target != background} of -log(1 - p_bg)

It only fires on pixels that must not be background, it is bounded (p_bg is clamped away
from 1), and it leaves the marginal prior of Background essentially untouched in clean
in-distribution batches, where p_bg is already small on foreground pixels.

``class_weights`` is wired up for real, defaulting to a deliberately *mild*
square-root-inverse-frequency schedule with a floor of 0.7, so no majority class is
suppressed hard.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

BACKGROUND_INDEX = 1
IGNORE_INDEX = 0


def inverse_frequency_weights(counts, mode: str = "mild", floor: float = 0.7, cap: float = 3.0):
    """Build per-class weights from pixel counts.

    ``counts`` is indexed by class id and must include the ignore class at index 0.

    ``mode='mild'``   -> ``(median_frequency / frequency) ** 0.5`` clamped to ``[floor, cap]``
    ``mode='median'`` -> plain median-frequency balancing, clamped to ``[floor, cap]``
    ``mode='none'``   -> all ones
    """
    counts = torch.as_tensor(counts, dtype=torch.float64).flatten()
    if mode == "none":
        return torch.ones(counts.numel(), dtype=torch.float32)
    if mode not in ("mild", "median"):
        raise ValueError(f"unknown class weight mode: {mode}")

    labelled = counts[1:].sum()
    if labelled <= 0:
        return torch.ones(counts.numel(), dtype=torch.float32)
    frequency = counts / labelled
    median = torch.median(frequency[1:])
    safe = frequency.clamp_min(torch.finfo(torch.float64).tiny)
    weights = median / safe
    if mode == "mild":
        weights = weights.sqrt()
    weights = weights.clamp(min=floor, max=cap)
    weights[IGNORE_INDEX] = 1.0
    weights[BACKGROUND_INDEX] = max(float(weights[BACKGROUND_INDEX]), floor)
    return weights.to(torch.float32)


def bgfix_loss(
    output,
    target,
    class_weights=None,
    focal_gamma: float = 1.5,
    ce_weight: float = 0.55,
    dice_weight: float = 0.30,
    boundary_weight: float = 0.10,
    rare_weight: float = 0.5,
    rare_classes=(5, 7, 8),
    false_bg_weight: float = 0.25,
    background_index: int = BACKGROUND_INDEX,
    ignore_index: int = IGNORE_INDEX,
    return_components: bool = False,
):
    logits = output["logits"]
    classes = logits.shape[1]

    raw_ce = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )
    probability = logits.softmax(1)
    valid = (target != ignore_index) & (target >= 0) & (target < classes)
    safe_target = target.clamp(0, classes - 1)
    focal = (1 - probability.gather(1, safe_target[:, None]).squeeze(1)).pow(focal_gamma)
    ce = (raw_ce[valid] * focal[valid]).mean() if valid.any() else logits.new_zeros(())

    dice_values = []
    for cls in range(1, classes):
        actual = (target == cls) & valid
        if actual.any():
            pred = probability[:, cls][valid]
            truth = actual[valid].to(pred.dtype)
            dice_values.append(1 - (2 * (pred * truth).sum() + 1) / (pred.sum() + truth.sum() + 1))
    dice = torch.stack(dice_values).mean() if dice_values else logits.new_zeros(())

    rare = logits.new_zeros(())
    for cls in rare_classes:
        actual = (target == cls) & valid
        if actual.any():
            rare = rare + (-probability[:, cls][actual].clamp_min(1e-6).log()).mean()

    boundary_target = torch.zeros_like(target, dtype=torch.bool)
    boundary_target[:, 1:] |= target[:, 1:] != target[:, :-1]
    boundary_target[:, :, 1:] |= target[:, :, 1:] != target[:, :, :-1]
    bce = F.binary_cross_entropy_with_logits(
        output["boundary"].squeeze(1), boundary_target.float(), reduction="none"
    )

    false_bg = logits.new_zeros(())
    foreground = valid & (target != background_index)
    if false_bg_weight > 0 and foreground.any():
        p_background = probability[:, background_index][foreground]
        # -log(1 - p_bg); bounded because p_bg is kept away from 1.
        false_bg = (-torch.log1p(-p_background.clamp(max=1.0 - 1e-6))).mean()

    boundary = bce[valid].mean() if valid.any() else logits.new_zeros(())

    total = (
        ce_weight * ce
        + dice_weight * dice
        + rare_weight * 0.05 * rare
        + boundary_weight * boundary
        + false_bg_weight * false_bg
    )
    total = torch.nan_to_num(total, nan=0.0)
    if not return_components:
        return total
    return total, {
        "loss/total": float(total.detach()),
        "loss/focal_ce": float(ce.detach()),
        "loss/dice": float(dice.detach()),
        "loss/rare": float(rare.detach()),
        "loss/boundary": float(boundary.detach()),
        "loss/false_bg": float(false_bg.detach()),
    }


@torch.no_grad()
def confusion_matrix(logits, target, classes: int = 9, ignore_index: int = IGNORE_INDEX):
    """Accumulate a ``classes x classes`` confusion matrix; rows are ground truth."""
    prediction = logits.argmax(1)
    valid = (target != ignore_index) & (target >= 0) & (target < classes)
    encoded = target[valid] * classes + prediction[valid]
    return torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def metrics_from_confusion(confusion, background_index: int = BACKGROUND_INDEX):
    """mIoU over classes 1..C-1 plus the two Background error rates we care about."""
    confusion = confusion.to(torch.float64)
    classes = confusion.shape[0]
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    per_class = {
        str(cls): float(intersection[cls] / union[cls]) if union[cls] > 0 else float("nan")
        for cls in range(1, classes)
    }
    present = [cls for cls in range(1, classes) if union[cls] > 0 and not math.isnan(per_class[str(cls)])]
    miou = float(sum(per_class[str(cls)] for cls in present) / len(present)) if present else 0.0

    background_truth = confusion.sum(1)[background_index]
    background_pred = confusion.sum(0)[background_index]
    true_background = confusion[background_index, background_index]
    # predicted Background where the label is something else
    false_positive = float((background_pred - true_background) / max(1.0, confusion.sum() - background_truth))
    # labelled Background predicted as something else
    false_negative = float(
        (background_truth - true_background) / max(1.0, background_truth)
    )
    return {
        "mIoU": miou,
        "per_class_iou": per_class,
        "classes_in_mean": [int(cls) for cls in present],
        "background_false_positive_rate": false_positive,
        "background_false_negative_rate": false_negative,
        "confusion": confusion.to(torch.int64).cpu().tolist(),
    }
