# dataset settings
dataset_type = 'APPLESDataset'
img_norm_cfg = dict(
    mean=[2.44154813, 3.51976852, 1.69956677],
    std=[5.76571964, 6.02998482, 4.26762906],
    to_rgb=False)
crop_size = (512, 512)
gta_train_pipeline = None
cityscapes_train_pipeline = None
synthia_train_pipeline = [
    dict(type='LoadImageFromNumpy'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 760)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromNumpy'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromNumpy'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='APPLESDataset',
            data_root='dataset/good ones/512_crop/train/',
            img_dir='images',
            ann_dir='annotations',
            pipeline=synthia_train_pipeline),
        target=dict(
            type='APPLESDataset',
            data_root='dataset/bad ones/512_crop/train/',
            img_dir='images',
            ann_dir='annotations',
            pipeline=cityscapes_train_pipeline)),
    val=dict(
        type='APPLESDataset',
        data_root='dataset/good ones/512_crop/val/',
        img_dir='images',
        ann_dir='annotations',
        pipeline=test_pipeline),
    test=dict(
        type='APPLESDataset',
        data_root='dataset/bad ones/512_crop/test/',
        img_dir='images',
        ann_dir='annotations',
        pipeline=test_pipeline))
