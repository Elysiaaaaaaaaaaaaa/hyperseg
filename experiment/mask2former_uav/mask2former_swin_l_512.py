"""Mask2Former with a Swin-L (IN-22K, 384) backbone on the eight valid UAV classes.

This is the H2 competition-baseline arm: same dataset, label contract,
512x512 crop protocol, optimizer recipe and evaluation as H1
(``mask2former_mit_b3_512.py``), with only the backbone swapped to the
official MMSegmentation Swin-L configuration from
``mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py``.
The official ADE20K config crops at 640x640; H2 keeps 512x512 to stay
protocol-comparable with H1/H3 and to fit a single RTX 4090 in FP32.

Raw masks use 0 as Ignore and 1..8 as semantic classes.  BaseSegDataset and
LoadAnnotations reduce them to 255 and 0..7 respectively during training.
Paths can be overridden through the HYPERSEG_* environment variables used by
``run.py`` in this directory.
"""

_base_ = [
    '../../../mmsegmentation/configs/mask2former/'
    'mask2former_r50_8xb2-160k_ade20k-512x512.py'
]

project_root = '{{$HYPERSEG_PROJECT_ROOT:.}}'
data_root = '{{$HYPERSEG_DATA_ROOT:dataset}}'
split_dir = '{{$HYPERSEG_SPLIT_DIR:runs/splits}}'
pretrained_checkpoint = '{{$HYPERSEG_SWIN_L_CHECKPOINT:https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth}}'

crop_size = (512, 512)
num_classes = 8
class_names = (
    'Background',
    'Building',
    'Road',
    'Water',
    'Barren',
    'Vegetation',
    'Agricultural',
    'Vehicle',
)
palette = [
    [128, 128, 128],
    [220, 20, 60],
    [255, 255, 255],
    [0, 0, 255],
    [160, 82, 45],
    [0, 128, 0],
    [255, 215, 0],
    [255, 0, 255],
]
metainfo = dict(classes=class_names, palette=palette)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32),
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_checkpoint),
    ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        strides=[4, 8, 16, 32],
        num_classes=num_classes,
        ignore_index=255,
        # Includes the no-object class as the final entry.
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1]),
    ),
    test_cfg=dict(
        _delete_=True,
        mode='whole',
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 1.5),
        keep_ratio=True,
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.9),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs'),
]

dataset_common = dict(
    type='BaseSegDataset',
    img_suffix='.png',
    seg_map_suffix='.png',
    reduce_zero_label=True,
    metainfo=metainfo,
    data_prefix=dict(
        img_path=f'{data_root}/train/images',
        seg_map_path=f'{data_root}/train/masks',
    ),
)

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        **dataset_common,
        ann_file=f'{split_dir}/train.txt',
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        **dataset_common,
        ann_file=f'{split_dir}/val.txt',
        pipeline=test_pipeline,
        test_mode=True,
    ),
)
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        **dataset_common,
        ann_file=f'{split_dir}/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(
    type='IoUMetric', ignore_index=255, iou_metrics=['mIoU'])
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, weight_decay=0.05, eps=1e-8,
        betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    accumulative_counts=1,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
            'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
        },
        norm_decay_mult=0.0,
    ),
)
param_scheduler = [
    dict(
        type='PolyLR', eta_min=0.0, power=0.9,
        begin=0, end=20000, by_epoch=False),
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)
randomness = dict(seed=3407, deterministic=False)
work_dir = f'{project_root}/runs/mask2former_uav/mask2former_swin_l_512'
