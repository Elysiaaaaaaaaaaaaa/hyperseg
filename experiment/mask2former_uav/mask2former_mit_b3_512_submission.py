"""Unlabelled 1024px test-set export config for Mask2Former."""

_base_ = ['./mask2former_mit_b3_512.py']

test_image_dir = '{{$HYPERSEG_TEST_IMAGE_DIR:dataset/low_altitude_2026/images}}'

submission_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackSegInputs'),
]
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_prefix=dict(img_path=test_image_dir),
        img_suffix='.png',
        seg_map_suffix='.png',
        reduce_zero_label=True,
        metainfo=dict(
            classes=(
                'Background', 'Building', 'Road', 'Water', 'Barren',
                'Vegetation', 'Agricultural', 'Vehicle'),
            palette=[
                [128, 128, 128], [220, 20, 60], [255, 255, 255],
                [0, 0, 255], [160, 82, 45], [0, 128, 0],
                [255, 215, 0], [255, 0, 255],
            ],
        ),
        pipeline=submission_pipeline,
        test_mode=True,
    ),
)
test_evaluator = dict(
    type='IoUMetric', format_only=True, ignore_index=255,
    iou_metrics=['mIoU'])
