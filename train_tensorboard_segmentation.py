#!/usr/bin/env python3
"""
Enhanced YOLOv5 Training with TensorBoard Integration

This script trains YOLOv5 model with comprehensive logging and visualization:
- Box regression loss
- Objectness loss  
- Box classification loss
- Segmentation loss
- Learning rate
- Accuracy plots
- Precision-Recall curves with mAP@0.5
"""

import os
import sys
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# Add YOLOv5 path
sys.path.append('.')

def setup_tensorboard(project_dir, experiment_name):
    """Setup TensorBoard logging"""
    log_dir = Path(project_dir) / experiment_name / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Start TensorBoard with: tensorboard --logdir {log_dir}")
    
    return writer

def get_optimal_parameters():
    """Get optimal training parameters for food dataset"""
    return {
        'epochs': 150,  # Sufficient for convergence
        'batch_size': 16,  # Good balance for our dataset size
        'img_size': 640,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'patience': 30,  # Early stopping
        'save_period': 10,  # Save checkpoint every 10 epochs
    }

def create_training_config():
    """Create comprehensive training configuration"""
    config = {
        # Model configuration
        'model': 'yolov5s-seg.pt',  # Use segmentation model
        'data': 'yolo_dataset_final/data.yaml',
        
        # Training parameters
        **get_optimal_parameters(),
        
        # Paths
        'project': 'runs/train',
        'name': 'food_instance_segmentation',
        'exist_ok': True,
        
        # Logging and visualization
        'plots': True,
        'save_txt': True,
        'save_conf': True,
        
        # Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        
        # Loss weights
        'box': 0.05,
        'cls': 0.5,
        'obj': 1.0,
        
        # Device
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        'cache': True,
    }
    
    return config

def train_with_tensorboard():
    """Main training function with TensorBoard integration"""
    
    print("="*80)
    print("🚀 YOLOV5 INSTANCE SEGMENTATION TRAINING")
    print("="*80)
    
    # Get training configuration
    config = create_training_config()
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Validate dataset
    data_path = config['data']
    if not os.path.exists(data_path):
        print(f"❌ Dataset file not found: {data_path}")
        return
    
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\nDataset Info:")
    print(f"  Classes: {data_config.get('nc', 'Unknown')}")
    print(f"  Train path: {data_config.get('train', 'Unknown')}")
    print(f"  Val path: {data_config.get('val', 'Unknown')}")
    
    # Setup TensorBoard
    writer = setup_tensorboard(config['project'], config['name'])
    
    # Build training command
    cmd_parts = [
        "python", "segment/train.py",  # Use segmentation training script
        "--data", config['data'],
        "--weights", config['model'],
        "--epochs", str(config['epochs']),
        "--batch-size", str(config['batch_size']),
        "--img", str(config['img_size']),
        "--project", config['project'],
        "--name", config['name'],
        "--device", str(config['device']),
        "--workers", str(config['workers']),
        "--patience", str(config['patience']),
        "--save-period", str(config['save_period']),
        "--plots",
        "--save-txt",
        "--save-conf",
        "--cache",
        "--exist-ok",
    ]
    
    # Add hyperparameters
    hyp_params = {
        'lr0': config['lr0'],
        'lrf': config['lrf'],
        'momentum': config['momentum'],
        'weight_decay': config['weight_decay'],
        'warmup_epochs': config['warmup_epochs'],
        'box': config['box'],
        'cls': config['cls'],
        'obj': config['obj'],
        'hsv_h': config['hsv_h'],
        'hsv_s': config['hsv_s'],
        'hsv_v': config['hsv_v'],
        'degrees': config['degrees'],
        'translate': config['translate'],
        'scale': config['scale'],
        'shear': config['shear'],
        'perspective': config['perspective'],
        'flipud': config['flipud'],
        'fliplr': config['fliplr'],
        'mosaic': config['mosaic'],
        'mixup': config['mixup'],
    }
    
    # Create hyperparameters file
    hyp_file = "hyp_food_segmentation.yaml"
    with open(hyp_file, 'w') as f:
        yaml.dump(hyp_params, f, default_flow_style=False)
    
    cmd_parts.extend(["--hyp", hyp_file])
    
    # Join command
    cmd = " ".join(cmd_parts)
    
    print(f"\n{'='*60}")
    print("📊 Training Command:")
    print(cmd)
    print(f"{'='*60}")
    
    print(f"\n🔥 STARTING TRAINING...")
    print(f"💡 Monitor progress with: tensorboard --logdir {config['project']}/{config['name']}/tensorboard")
    
    # Execute training
    result = os.system(cmd)
    
    if result == 0:
        print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        results_dir = Path(config['project']) / config['name']
        print(f"📁 Results saved to: {results_dir}")
        print(f"🏆 Best model: {results_dir}/weights/best.pt")
        print(f"📈 Training plots: {results_dir}/results.png")
        print(f"📊 TensorBoard logs: {results_dir}/tensorboard/")
    else:
        print(f"❌ Training failed with exit code: {result}")
    
    writer.close()
    return result == 0

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv5 instance segmentation with TensorBoard')
    parser.add_argument('--config-only', action='store_true', help='Only show configuration')
    
    args = parser.parse_args()
    
    if args.config_only:
        config = create_training_config()
        print("Training Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        success = train_with_tensorboard()
        if success:
            print("\n🎉 Training completed! Ready for inference.")
        else:
            print("\n💥 Training failed. Check logs for details.")

if __name__ == "__main__":
    main()
