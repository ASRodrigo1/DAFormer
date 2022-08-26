log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True

#mean=[ 2.44154813, 3.51976852, 1.69956677]
#std=[ 5.76571964, 6.02998482, 4.26762906]
num_channels = 5
mean_good=[ 3.83292175,  5.97356766 , 2.73248754, 14.58615832, 37.98090491,  0.51887011][:num_channels]
std_good=[ 6.93827298,  7.13972697,  5.68746431, 12.50132059 ,24.12176963 , 1.16183588][:num_channels]

mean_bad=[ 8.57299022,  9.39551486 , 5.38475858, 20.26364527 ,38.53171637 , 0.61854915][:num_channels]
std_bad=[ 6.7109087,   4.92003234,  4.16396892,  8.29271099, 13.68164234 , 1.22788755][:num_channels]

bands = [0, 1, 2, 3, 4, 5][:num_channels]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5_5bands.pth',
    backbone=dict(type='mit_b5', in_chans=num_channels, style='pytorch'),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=dict(type='BN', requires_grad=True))),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(
        work_dir='dataset/work_dirs/apples_good/segformer/0'),
    test_cfg=dict(mode='whole'))
dataset_type = 'APPLESDataset'
img_norm_cfg = dict(
    mean=mean_good,
    std=std_good,
    to_rgb=False)
crop_size = (256, 256)
gta_train_pipeline = None
cityscapes_train_pipeline = None
test_pipeline = [
    dict(type='LoadImageFromNumpy'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=mean_bad,
                std=std_bad,
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='APPLESDataset',
            data_root='dataset/good ones/',
            img_dir='512_crop/train/images',
            ann_dir='512_crop/train/annotations',
            pipeline=[
                dict(type='LoadImageFromNumpy', bands=bands, to_float32=True),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(512, 512),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(256, 256),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='Normalize',
                    mean=mean_good,
                    std=std_good,
                    to_rgb=False),
                dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ]),
        target=dict(
            type='APPLESDataset',
            data_root='dataset/bad ones',
            img_dir='512_crop/train/images',
            ann_dir='512_crop/train/annotations',
            pipeline=[
                dict(type='LoadImageFromNumpy', bands=bands, to_float32=True),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(512, 512),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(256, 256),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='Normalize',
                    mean=mean_bad,
                    std=std_bad,
                    to_rgb=False),
                dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ]),
        rare_class_sampling=None),
    val=dict(
        type='APPLESDataset',
        data_root='dataset/good ones/',
        img_dir='512_crop/val/images',
        ann_dir='512_crop/val/annotations',
        pipeline=[
            dict(type='LoadImageFromNumpy', bands=bands, to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=mean_good,
                        std=std_good,
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='APPLESDataset',
        data_root='dataset/bad ones/',
        img_dir='512_crop/test/images',
        ann_dir='512_crop/test/annotations',
        pipeline=[
            dict(type='LoadImageFromNumpy', bands=bands, to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=mean_bad,
                        std=std_bad,
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
uda = dict(
    type='DACS',
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=1000,
    print_grad_magnitude=False)
use_ddp_wrapper = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = None
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
seed = 0
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(
    by_epoch=False,
    interval=2000,
    max_keep_ckpts=1,
    meta=dict(
        CLASSES=('background', 'plantation'), PALETTE=[[0, 0, 0], [255, 0,
                                                                   0]]))
evaluation = dict(interval=500, metric=['mIoU', 'mFscore'])
name = 'uda_daformer_segformer'
exp = 'basic'
name_dataset = 'apples'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
source_train_pipeline = [
    dict(type='LoadImageFromNumpy', bands=bands, to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=mean_good,
        std=std_good,
        to_rgb=False),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
target_train_pipeline = [
    dict(type='LoadImageFromNumpy', bands=bands, to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=mean_bad,
        std=std_bad,
        to_rgb=False),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
val_pipeline = [
    dict(type='LoadImageFromNumpy', bands=bands, to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=mean_good,
                std=std_good,
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
work_dir = 'dataset/work_dirs/apples_good/segformer/0'
gpu_ids = range(0, 1)
