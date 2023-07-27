_base_ = './queryinst_r50_fpn_1x_coco.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
min_values = (480, 512, 544, 576, 608, 640) # 672, 704, 736, 768, 800)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        #img_scale=[(1333, value) for value in min_values],
        img_scale=[(480, 640)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

data = dict(train=dict(pipeline=train_pipeline))
lr_config = dict(policy='step', step=[27, 33])
total_epochs = 80
runner = dict(max_epochs=total_epochs)
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)

# load_from = 'C:/Users/Administrator/Downloads/swin_large_patch4_window7_224_22k.pth'
