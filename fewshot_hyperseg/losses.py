import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 9
IGNORE_INDEX = 0


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=0, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        c = logits.shape[1]
        valid = target != self.ignore_index
        safe = target.clone()
        safe[~valid] = 0
        prob = F.softmax(logits, 1)
        onehot = F.one_hot(safe.long(), c).permute(0,3,1,2).float()
        v = valid[:,None].float()
        prob, onehot = prob*v, onehot*v
        inter = (prob*onehot).sum((0,2,3))
        den = (prob+onehot).sum((0,2,3))
        dice = (2*inter+self.smooth)/(den+self.smooth)
        # class 0 is Ignore and is excluded.
        return 1 - dice[1:].mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=0):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target.long(), reduction="none",
                             ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        loss = ((1-pt)**self.gamma)*ce
        valid = target != self.ignore_index
        return loss[valid].mean() if valid.any() else logits.sum()*0


class BoundaryLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index

    @staticmethod
    def make_boundary(target, ignore_index=0):
        valid = target != ignore_index
        b = torch.zeros_like(target, dtype=torch.bool)

        dh = target[:,:,1:] != target[:,:,:-1]
        vh = valid[:,:,1:] & valid[:,:,:-1]
        b[:,:,1:] |= dh & vh
        b[:,:,:-1] |= dh & vh

        dv = target[:,1:,:] != target[:,:-1,:]
        vv = valid[:,1:,:] & valid[:,:-1,:]
        b[:,1:,:] |= dv & vv
        b[:,:-1,:] |= dv & vv

        # One-pixel dilation makes the boundary signal less sparse.
        b = F.max_pool2d(b.float()[:,None], 3, 1, 1)[:,0] > 0
        return b.float(), valid.float()

    def forward(self, boundary_logits, target):
        if boundary_logits.ndim == 4:
            boundary_logits = boundary_logits[:,0]
        bt, valid = self.make_boundary(target, self.ignore_index)

        bce_map = F.binary_cross_entropy_with_logits(
            boundary_logits, bt, reduction="none")
        bce = (bce_map*valid).sum()/valid.sum().clamp_min(1)

        p = torch.sigmoid(boundary_logits)*valid
        t = bt*valid
        inter = (p*t).flatten(1).sum(1)
        dice = 1-(2*inter+1)/(p.flatten(1).sum(1)+t.flatten(1).sum(1)+1)
        return bce+dice.mean(), {"boundary_bce":bce.detach(), "boundary_dice":dice.mean().detach()}


class PrototypeSeparationLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, prototypes):
        if prototypes.shape[1] < 2:
            return prototypes.sum()*0
        p = F.normalize(prototypes, dim=-1)
        sim = torch.bmm(p, p.transpose(1,2))
        n = sim.shape[1]
        eye = torch.eye(n, device=sim.device, dtype=torch.bool)[None]
        neg = sim.masked_select(~eye)
        return F.relu(neg-self.margin).mean()


class FusionEntropyRegularizer(nn.Module):
    def forward(self, weights):
        w = weights.clamp_min(1e-8)
        return -(w*torch.log(w)).sum(-1).mean()


class HyperSegLoss(nn.Module):
    """
    Total:
        L = L_CE + L_Dice + 0.25 L_Focal
          + lambda_boundary L_boundary
          + lambda_proto L_proto
          + lambda_fusion L_fusion
    """
    def __init__(
        self,
        ignore_index=0,
        lambda_boundary=0.10,
        lambda_proto=0.05,
        lambda_fusion=0.01,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.dice = DiceLoss(ignore_index)
        self.focal = FocalLoss(ignore_index=ignore_index)
        self.boundary = BoundaryLoss(ignore_index)
        self.proto = PrototypeSeparationLoss()
        self.fusion = FusionEntropyRegularizer()
        self.lambda_boundary = lambda_boundary
        self.lambda_proto = lambda_proto
        self.lambda_fusion = lambda_fusion

    def forward(self, outputs, target):
        logits = outputs["logits"]

        ce = F.cross_entropy(
            logits, target.long(), ignore_index=self.ignore_index)

        dice = self.dice(logits, target)
        focal = self.focal(logits, target)
        seg = ce + dice + 0.25*focal

        if "boundary" in outputs:
            lb, bd = self.boundary(outputs["boundary"], target)
        else:
            lb = logits.sum()*0
            bd = {"boundary_bce":lb.detach(), "boundary_dice":lb.detach()}

        lp = self.proto(outputs["prototypes"]) if "prototypes" in outputs else logits.sum()*0
        lf = self.fusion(outputs["fusion_weights"]) if "fusion_weights" in outputs else logits.sum()*0

        total = seg + self.lambda_boundary*lb + self.lambda_proto*lp + self.lambda_fusion*lf

        details = {
            "loss": total.detach(),
            "seg_loss": seg.detach(),
            "ce": ce.detach(),
            "dice": dice.detach(),
            "focal": focal.detach(),
            "boundary_loss": lb.detach(),
            "prototype_loss": lp.detach(),
            "fusion_loss": lf.detach(),
            **bd,
        }
        return total, details


def hyperseg_loss(outputs, target, criterion=None):
    criterion = criterion or HyperSegLoss()
    return criterion(outputs, target)


@torch.no_grad()
def mean_iou(logits, target, num_classes=9, ignore_index=0):
    pred = logits.argmax(1)
    per_class = {}
    vals = []
    for cls in range(1, num_classes):
        valid = target != ignore_index
        p = (pred == cls) & valid
        t = (target == cls) & valid
        inter = (p&t).sum().float()
        union = (p|t).sum().float()
        if union > 0:
            v = (inter/union).item()
            vals.append(v)
            per_class[cls] = v
        else:
            per_class[cls] = float("nan")
    return (sum(vals)/len(vals) if vals else 0.0), per_class