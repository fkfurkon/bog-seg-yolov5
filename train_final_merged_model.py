#!/usr/bin/env python3
"""
Final YOLO Training Script for Merged Dataset

This script trains a YOLOv5 model on the merged bounding box + segmentation dataset.
Includes proper configuration for food classification with multiple annotation types.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

def validate_dataset(data_yaml_path):
    """Validate the dataset configuration"""
    print(f"Validating dataset: {data_yaml_path}")
    
    if not os.path.exists(data_yaml_path):
        print(f"❌ Dataset file not found: {data_yaml_path}")
        return False
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_path = Path(data_config['path'])
    train_path = dataset_path / data_config['train']
    val_path = dataset_path / data_config['val']
    
    print(f"Dataset path: {dataset_path}")
    print(f"Train images: {train_path}")
    print(f"Val images: {val_path}")
    print(f"Number of classes: {data_config['nc']}")
    print(f"Classes: {list(data_config['names'].values())}")
    
    # Check if directories exist
    if not train_path.exists():
        print(f"❌ Training images directory not found: {train_path}")
        return False
    
    if not val_path.exists():
        print(f"❌ Validation images directory not found: {val_path}")
        return False
    
    # Count files
    train_images = len(list(train_path.glob("*")))
    val_images = len(list(val_path.glob("*")))
    
    train_labels_path = dataset_path / "labels/train"
    val_labels_path = dataset_path / "labels/val"
    
    train_labels = len(list(train_labels_path.glob("*.txt"))) if train_labels_path.exists() else 0
    val_labels = len(list(val_labels_path.glob("*.txt"))) if val_labels_path.exists() else 0
    
    print(f"Train: {train_images} images, {train_labels} labels")
    print(f"Val: {val_images} images, {val_labels} labels")
    
    if train_images == 0 or val_images == 0:
        print("❌ No images found in train or val directories")
        return False
    
    if train_labels == 0 or val_labels == 0:
        print("❌ No labels found in train or val directories")
        return False
    
    print("✅ Dataset validation passed")
    return True

def get_optimal_hyperparameters():
    """Get optimal hyperparameters for food classification"""
    return {
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 0.05,
        'cls': 0.5,
        'cls_pw': 1.0,
        'obj': 1.0,
        'obj_pw': 1.0,
        'iou_t': 0.20,
        'anchor_t': 4.0,
        'fl_gamma': 0.0,
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
        'copy_paste': 0.0
    }

def train_yolo_model(data_yaml, model_size='s', epochs=100, img_size=640, batch_size=16, 
                    project='runs/train', name='final_merged_model', resume=False):
    """Train YOLOv5 model on the merged dataset"""
    
    print(f"\n🚀 Starting YOLO training...")
    print(f"Model: YOLOv5{model_size}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch_size}")
    
    # Validate dataset first
    if not validate_dataset(data_yaml):
        print("❌ Dataset validation failed. Training aborted.")
        return
    
    # Build training command
    cmd_parts = [
        "python", "train.py",
        "--data", str(data_yaml),
        "--weights", f"yolov5{model_size}.pt",
        "--epochs", str(epochs),
        "--img", str(img_size),
        "--batch-size", str(batch_size),
        "--project", project,
        "--name", name,
        "--cache",
        "--device", "0"  # Use GPU if available
    ]
    
    if resume:
        cmd_parts.extend(["--resume"])
    
    # Add hyperparameters
    hyp_params = get_optimal_hyperparameters()
    for key, value in hyp_params.items():
        cmd_parts.extend([f"--{key}", str(value)])
    
    # Create hyperparameters file
    hyp_file = f"hyp_food_classification.yaml"
    with open(hyp_file, 'w') as f:
        yaml.dump(hyp_params, f, default_flow_style=False)
    
    print(f"Hyperparameters saved to: {hyp_file}")
    
    # Join command
    cmd = " ".join(cmd_parts)
    print(f"\nTraining command:")
    print(cmd)
    
    # Execute training
    print(f"\n{'='*60}")
    print("🔥 STARTING TRAINING 🔥")
    print(f"{'='*60}")
    
    os.system(cmd)
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {project}/{name}")

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv5 on merged food dataset')
    parser.add_argument('--data', 
                       default='yolo_dataset_final/data.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--model', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       default='s',
                       help='Model size (nano, small, medium, large, xlarge)')
    parser.add_argument('--epochs', 
                       type=int, 
                       default=100,
                       help='Number of training epochs')
    parser.add_argument('--img-size', 
                       type=int, 
                       default=640,
                       help='Input image size')
    parser.add_argument('--batch-size', 
                       type=int, 
                       default=16,
                       help='Batch size')
    parser.add_argument('--project', 
                       default='runs/train',
                       help='Project directory')
    parser.add_argument('--name', 
                       default='final_merged_food_model',
                       help='Experiment name')
    parser.add_argument('--resume', 
                       action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--validate-only', 
                       action='store_true',
                       help='Only validate dataset without training')
    
    args = parser.parse_args()
    
    if args.validate_only:
        validate_dataset(args.data)
    else:
        train_yolo_model(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            project=args.project,
            name=args.name,
            resume=args.resume
        )

if __name__ == "__main__":
    main()
