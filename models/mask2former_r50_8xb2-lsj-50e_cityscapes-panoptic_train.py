auto_scale_lr = dict(base_batch_size=16, enable=False)  # base_batch_size=8
backend_args = None
batch_augments = [
    dict(
        img_pad_value=0,
        mask_pad_value=0,
        pad_mask=True,
        pad_seg=True,
        seg_pad_value=255,
        size=(
            1024,
            1024,
        ),
        type="BatchFixedSizePad",
    ),
]
data_preprocessor = dict(
    batch_augments=[
        dict(
            img_pad_value=0,
            mask_pad_value=0,
            pad_mask=True,
            pad_seg=True,
            seg_pad_value=255,
            size=(
                1024,
                1024,
            ),
            type="BatchFixedSizePad",
        ),
    ],
    bgr_to_rgb=True,
    mask_pad_value=0,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_mask=True,
    pad_seg=True,
    pad_size_divisor=32,
    seg_pad_value=255,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type="DetDataPreprocessor",
)
data_root = "./data/cityscapes/"
dataset_type = "CityscapesPanopticDataset"
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=5000, max_keep_ckpts=3, save_last=True, type="CheckpointHook"),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(type="DetVisualizationHook"),
)
default_scope = "mmdet"
dynamic_intervals = [
    (
        365001,
        368750,
    ),
]
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
env_cfg = dict(
    cudnn_benchmark=False, dist_cfg=dict(backend="nccl"), mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0)
)
image_size = (
    1024,
    1024,
)
interval = 5000
load_from = None
log_level = "INFO"
log_processor = dict(by_epoch=False, type="LogProcessor", window_size=50)
max_iters = 368750
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=-1,
        init_cfg=dict(checkpoint="torchvision://resnet50", type="Pretrained"),
        norm_cfg=dict(requires_grad=False, type="BN"),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style="pytorch",
        type="ResNet",
    ),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                img_pad_value=0,
                mask_pad_value=0,
                pad_mask=True,
                pad_seg=True,
                seg_pad_value=255,
                size=(
                    1024,
                    1024,
                ),
                type="BatchFixedSizePad",
            ),
        ],
        bgr_to_rgb=True,
        mask_pad_value=0,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_seg=True,
        pad_size_divisor=32,
        seg_pad_value=255,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type="DetDataPreprocessor",
    ),
    init_cfg=None,
    panoptic_fusion_head=dict(
        init_cfg=None,
        loss_panoptic=None,
        num_stuff_classes=11,  # 53
        num_things_classes=8,  # 80
        type="MaskFormerFusionHead",
    ),
    panoptic_head=dict(
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        loss_cls=dict(
            class_weight=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.1,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 1.0,
                # 0.1,
            ],
            loss_weight=2.0,
            reduction="mean",
            type="CrossEntropyLoss",
            use_sigmoid=False,
        ),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction="mean",
            type="DiceLoss",
            use_sigmoid=True,
        ),
        loss_mask=dict(loss_weight=5.0, reduction="mean", type="CrossEntropyLoss", use_sigmoid=True),
        num_queries=100,
        num_stuff_classes=11,  # 53
        num_things_classes=8,  # 80
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type="ReLU"),
            encoder=dict(
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type="ReLU"),
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        num_fcs=2,
                    ),
                    self_attn_cfg=dict(
                        batch_first=True, dropout=0.0, embed_dims=256, num_heads=8, num_levels=3, num_points=4
                    ),
                ),
                num_layers=6,
            ),
            norm_cfg=dict(num_groups=32, type="GN"),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            type="MSDeformAttnPixelDecoder",
        ),
        positional_encoding=dict(normalize=True, num_feats=128),
        strides=[
            4,
            8,
            16,
            32,
        ],
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(batch_first=True, dropout=0.0, embed_dims=256, num_heads=8),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2,
                ),
                self_attn_cfg=dict(batch_first=True, dropout=0.0, embed_dims=256, num_heads=8),
            ),
            num_layers=9,
            return_intermediate=True,
        ),
        type="Mask2FormerHead",
    ),
    test_cfg=dict(
        filter_low_score=True, instance_on=True, iou_thr=0.8, max_per_image=100, panoptic_on=True, semantic_on=False
    ),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type="ClassificationCost", weight=2.0),
                dict(type="CrossEntropyLossCost", use_sigmoid=True, weight=5.0),
                dict(eps=1.0, pred_act=True, type="DiceCost", weight=5.0),
            ],
            type="HungarianAssigner",
        ),
        importance_sample_ratio=0.75,
        num_points=12544,
        oversample_ratio=3.0,
        sampler=dict(type="MaskPseudoSampler"),
    ),
    type="Mask2Former",
)
num_classes = 19  # 133
num_stuff_classes = 11  # 53
num_things_classes = 8  # 80
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type="AdamW",
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(decay_mult=1.0, lr_mult=0.1),
            level_embed=dict(decay_mult=0.0, lr_mult=1.0),
            query_embed=dict(decay_mult=0.0, lr_mult=1.0),
            query_feat=dict(decay_mult=0.0, lr_mult=1.0),
        ),
        norm_decay_mult=0.0,
    ),
    type="OptimWrapper",
)
param_scheduler = dict(
    begin=0,
    by_epoch=False,
    end=368750,
    gamma=0.1,
    milestones=[
        327778,
        355092,
    ],
    type="MultiStepLR",
)
resume = False
test_cfg = dict(type="TestLoop")
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file="annotations/cityscapes_panoptic_val.json",
        backend_args=None,
        data_prefix=dict(img="leftImg8bit/cityscapes_panoptic_val/", seg="gtFine/cityscapes_panoptic_val/"),
        data_root="./data/cityscapes/",
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type="Resize",
            ),
            dict(backend_args=None, type="LoadPanopticAnnotations"),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CityscapesPanopticDataset",
    ),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
test_evaluator = [
    dict(
        ann_file="./data/cityscapes/annotations/cityscapes_panoptic_val.json",
        backend_args=None,
        seg_prefix="./data/cityscapes/gtFine/cityscapes_panoptic_val",
        type="CocoPanopticMetric",
    ),
    # dict(
    #     ann_file='data/cityscapes/annotations/cityscapes_instances_val.json',
    #     backend_args=None,
    #     metric=[
    #         'bbox',
    #         'segm',
    #     ],
    #     type='CocoMetric'),
]
test_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(
        keep_ratio=True,
        scale=(
            1333,
            800,
        ),
        type="Resize",
    ),
    dict(backend_args=None, type="LoadPanopticAnnotations"),
    dict(
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
        type="PackDetInputs",
    ),
]
train_cfg = dict(
    dynamic_intervals=[
        (
            365001,
            368750,
        ),
    ],
    max_iters=368750,  # 368750
    type="IterBasedTrainLoop",
    val_interval=5000,
)
train_dataloader = dict(
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    batch_size=2,  # 2
    dataset=dict(
        ann_file="annotations/cityscapes_panoptic_train.json",
        backend_args=None,
        data_prefix=dict(img="leftImg8bit/cityscapes_panoptic_train/", seg="gtFine/cityscapes_panoptic_train/"),
        data_root="./data/cityscapes/",
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, to_float32=True, type="LoadImageFromFile"),
            dict(backend_args=None, type="LoadPanopticAnnotations", with_bbox=True, with_mask=True, with_seg=True),
            dict(prob=0.5, type="RandomFlip"),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type="RandomResize",
            ),
            dict(
                allow_negative_crop=True,
                crop_size=(
                    1024,
                    1024,
                ),
                crop_type="absolute",
                recompute_bbox=True,
                type="RandomCrop",
            ),
            dict(type="PackDetInputs"),
        ],
        type="CityscapesPanopticDataset",
    ),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)
train_pipeline = [
    dict(backend_args=None, to_float32=True, type="LoadImageFromFile"),
    dict(backend_args=None, type="LoadPanopticAnnotations", with_bbox=True, with_mask=True, with_seg=True),
    dict(prob=0.5, type="RandomFlip"),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            1024,
            1024,
        ),
        type="RandomResize",
    ),
    dict(
        allow_negative_crop=True,
        crop_size=(
            1024,
            1024,
        ),
        crop_type="absolute",
        recompute_bbox=True,
        type="RandomCrop",
    ),
    dict(type="PackDetInputs"),
]
val_cfg = dict(type="ValLoop")
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file="annotations/cityscapes_panoptic_val.json",
        backend_args=None,
        data_prefix=dict(img="leftImg8bit/cityscapes_panoptic_val/", seg="gtFine/cityscapes_panoptic_val/"),
        data_root="./data/cityscapes/",
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type="Resize",
            ),
            dict(backend_args=None, type="LoadPanopticAnnotations"),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        type="CityscapesPanopticDataset",
    ),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
val_evaluator = [
    dict(
        ann_file="./data/cityscapes/annotations/cityscapes_panoptic_val.json",
        backend_args=None,
        seg_prefix="./data/cityscapes/gtFine/cityscapes_panoptic_val/",
        type="CocoPanopticMetric",
    ),
    # dict(
    #     ann_file='data/cityscapes/annotations/cityscapes_instances_val.json',
    #     backend_args=None,
    #     metric=[
    #         'bbox',
    #         'segm',
    #     ],
    #     type='CocoMetric'),
]
vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
)
