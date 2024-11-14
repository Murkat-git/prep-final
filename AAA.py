import os.path as osp
import mmcv
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv import Config
from mmdet.utils import get_device

def prepare_config():
    """Prepare custom configuration for Co-DETR."""
    # Base config from Co-DETR
    cfg = Config.fromfile('configs/co_detr/co_detr_r50_8x2_50e_coco.py')
    
    # Modify dataset settings
    cfg.dataset_type = 'CustomDataset'
    cfg.data_root = 'path/to/your/dataset/'
    
    # Training and validation dataset settings
    cfg.data.train.type = 'CustomDataset'
    cfg.data.train.ann_file = 'path/to/your/train/annotations.json'
    cfg.data.train.img_prefix = 'path/to/your/train/images/'
    
    cfg.data.val.type = 'CustomDataset'
    cfg.data.val.ann_file = 'path/to/your/val/annotations.json'
    cfg.data.val.img_prefix = 'path/to/your/val/images/'
    
    cfg.data.test.type = 'CustomDataset'
    cfg.data.test.ann_file = 'path/to/your/test/annotations.json'
    cfg.data.test.img_prefix = 'path/to/your/test/images/'
    
    # Modify number of classes based on your dataset
    cfg.model.bbox_head.num_classes = 10  # Change this to your number of classes
    
    # Training settings
    cfg.optimizer = dict(type='AdamW', lr=0.0001)
    cfg.optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    cfg.lr_config = dict(
        policy='step',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=0.001,
        step=[27, 33])
    
    # Runtime settings
    cfg.total_epochs = 50
    cfg.checkpoint_config = dict(interval=5)
    cfg.log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
    cfg.evaluation = dict(interval=5)
    
    # Set working directory
    cfg.work_dir = './work_dirs/co_detr_custom'
    
    return cfg

def prepare_dataset_config(cfg):
    """Prepare dataset pipeline configuration."""
    # Training pipeline
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **cfg.img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    ]
    
    # Test pipeline
    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1333, 800),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    
    return cfg

def main():
    # Get configuration
    cfg = prepare_config()
    cfg = prepare_dataset_config(cfg)
    
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]
    
    # Build model
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    
    # Set device
    device = get_device()
    model.to(device)
    
    # Create work directory
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # Training
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True)

if __name__ == '__main__':
    main()