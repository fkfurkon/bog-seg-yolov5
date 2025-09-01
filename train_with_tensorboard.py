#!/usr/bin/env python
"""
YOLOv5 Segmentation Training with TensorBoard Logging
This script trains a YOLOv5 segmentation model with comprehensive logging for:
- Box regression loss
- Objectness loss
- Classification loss
- Segmentation loss
- Learning rate
- Accuracy plots
- Precision-Recall curves with mAP@0.5
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add YOLOv5 paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from segment.train import main as segment_train_main
from segment.val import main as segment_val_main
from utils.general import LOGGER, colorstr, increment_path
from utils.plots import plot_results
from utils.torch_utils import select_device

def create_tensorboard_logger(log_dir):
    """Create TensorBoard logger"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    return writer

def log_training_metrics(writer, metrics, epoch):
    """Log training metrics to TensorBoard"""
    if metrics:
        # Box regression loss
        if 'box_loss' in metrics:
            writer.add_scalar('Loss/Box_Regression', metrics['box_loss'], epoch)
        
        # Objectness loss
        if 'obj_loss' in metrics:
            writer.add_scalar('Loss/Objectness', metrics['obj_loss'], epoch)
        
        # Classification loss
        if 'cls_loss' in metrics:
            writer.add_scalar('Loss/Classification', metrics['cls_loss'], epoch)
        
        # Segmentation loss
        if 'seg_loss' in metrics:
            writer.add_scalar('Loss/Segmentation', metrics['seg_loss'], epoch)
        
        # Learning rate
        if 'lr' in metrics:
            writer.add_scalar('Learning_Rate/LR', metrics['lr'], epoch)
        
        # Validation metrics
        if 'precision' in metrics:
            writer.add_scalar('Metrics/Precision', metrics['precision'], epoch)
        
        if 'recall' in metrics:
            writer.add_scalar('Metrics/Recall', metrics['recall'], epoch)
        
        if 'mAP_0.5' in metrics:
            writer.add_scalar('Metrics/mAP@0.5', metrics['mAP_0.5'], epoch)
        
        if 'mAP_0.5:0.95' in metrics:
            writer.add_scalar('Metrics/mAP@0.5:0.95', metrics['mAP_0.5:0.95'], epoch)

def create_precision_recall_plot(precision, recall, ap, save_path):
    """Create Precision-Recall curve plot"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot PR curve
    ax.plot(recall, precision, linewidth=2, label=f'mAP@0.5 = {ap:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv5 Segmentation with TensorBoard')
    parser.add_argument('--weights', type=str, default='yolov5s-seg.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='food_dataset.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train-seg', help='save to project/name')
    parser.add_argument('--name', default='food_segmentation', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='*', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    
    # Segmentation specific args
    parser.add_argument('--mask-ratio', type=int, default=4, help='Downsample the truth masks to saving memory')
    parser.add_argument('--no-overlap', action='store_true', help='Overlap masks train faster at slightly less mAP')
    
    args = parser.parse_args()
    
    # Setup paths
    save_dir = Path(args.project) / args.name
    log_dir = save_dir / 'tensorboard'
    
    # Create TensorBoard logger
    writer = create_tensorboard_logger(log_dir)
    
    LOGGER.info(f'{colorstr("TensorBoard:")} Starting logging to {log_dir}')
    LOGGER.info(f'{colorstr("TensorBoard:")} Run "tensorboard --logdir={log_dir}" to view logs')
    
    try:
        # Start training
        LOGGER.info(f'{colorstr("Training:")} Starting YOLOv5 segmentation training...')
        
        # Prepare arguments for segment training
        train_args = [
            '--weights', str(args.weights),
            '--data', str(args.data),
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--imgsz', str(args.imgsz),
            '--project', str(args.project),
            '--name', str(args.name),
            '--device', str(args.device),
            '--cache', str(args.cache) if args.cache else 'ram',
            '--workers', str(args.workers),
            '--optimizer', str(args.optimizer),
            '--mask-ratio', str(args.mask_ratio),
        ]
        
        if args.rect:
            train_args.append('--rect')
        if args.resume:
            train_args.extend(['--resume', str(args.resume) if isinstance(args.resume, str) else ''])
        if args.nosave:
            train_args.append('--nosave')
        if args.noval:
            train_args.append('--noval')
        if args.noautoanchor:
            train_args.append('--noautoanchor')
        if args.multi_scale:
            train_args.append('--multi-scale')
        if args.single_cls:
            train_args.append('--single-cls')
        if args.sync_bn:
            train_args.append('--sync-bn')
        if args.exist_ok:
            train_args.append('--exist-ok')
        if args.quad:
            train_args.append('--quad')
        if args.cos_lr:
            train_args.append('--cos-lr')
        if args.no_overlap:
            train_args.append('--no-overlap')
        
        # Add additional args
        if args.hyp:
            train_args.extend(['--hyp', str(args.hyp)])
        if args.cfg:
            train_args.extend(['--cfg', str(args.cfg)])
        if args.label_smoothing:
            train_args.extend(['--label-smoothing', str(args.label_smoothing)])
        if args.patience:
            train_args.extend(['--patience', str(args.patience)])
        if args.save_period > 0:
            train_args.extend(['--save-period', str(args.save_period)])
        if args.seed:
            train_args.extend(['--seed', str(args.seed)])
        if args.freeze:
            train_args.extend(['--freeze'] + [str(x) for x in args.freeze])
        
        # Train the model
        import sys
        old_argv = sys.argv
        sys.argv = ['segment/train.py'] + train_args
        
        try:
            from segment.train import main as train_main
            train_main()
        finally:
            sys.argv = old_argv
        
        LOGGER.info(f'{colorstr("Training:")} Training completed successfully!')
        
        # Run validation to get final metrics
        LOGGER.info(f'{colorstr("Validation:")} Running final validation...')
        
        val_args = [
            '--weights', str(save_dir / 'weights' / 'best.pt'),
            '--data', str(args.data),
            '--batch-size', str(args.batch_size),
            '--imgsz', str(args.imgsz),
            '--task', 'val',
            '--device', str(args.device),
            '--save-txt', '--save-conf',
            '--plots'
        ]
        
        old_argv = sys.argv
        sys.argv = ['segment/val.py'] + val_args
        
        try:
            from segment.val import main as val_main
            val_main()
        finally:
            sys.argv = old_argv
        
        LOGGER.info(f'{colorstr("Validation:")} Validation completed!')
        
    except Exception as e:
        LOGGER.error(f'{colorstr("Error:")} Training failed: {e}')
        raise
    finally:
        writer.close()
    
    LOGGER.info(f'{colorstr("Complete:")} Training and logging completed!')
    LOGGER.info(f'{colorstr("TensorBoard:")} View logs with: tensorboard --logdir={log_dir}')

if __name__ == '__main__':
    main()
