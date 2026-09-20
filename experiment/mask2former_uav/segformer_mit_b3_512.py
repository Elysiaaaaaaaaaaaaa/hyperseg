"""MMSeg SegFormer control using the same MiT-B3/data/training protocol."""

_base_ = ['./mask2former_mit_b3_512.py']

model = dict(
    decode_head=dict(
        _delete_=True,
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        ignore_index=255,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
)

work_dir = 'runs/mask2former_uav/segformer_mit_b3_512'

