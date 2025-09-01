#!/usr/bin/env python
"""
YOLOv5 Segmentation Training - Standard YOLO Format
This script trains YOLOv5 segmentation model using proper YOLO dataset format.
"""

import os
import sys
from pathlib import Path

import yaml

# Add YOLOv5 to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def train_yolo_segmentation():
    """Train YOLOv5 segmentation model with proper YOLO format."""
    print("🚀 Starting YOLOv5 Segmentation Training")
    print("=" * 60)

    # Dataset configuration
    data_config = "yolo_dataset/data.yaml"

    # Check if dataset exists
    if not os.path.exists(data_config):
        print("❌ Error: Dataset not found!")
        print("Run organize_yolo_dataset.py first to create proper YOLO format")
        return

    # Load dataset info
    with open(data_config) as f:
        data = yaml.safe_load(f)

    print("📊 Dataset Configuration:")
    print(f"   Path: {data['path']}")
    print(f"   Classes: {data['nc']}")
    print(f"   Train: {data['train']}")
    print(f"   Val: {data['val']}")
    print(f"   Names: {list(data['names'].values())}")

    # Training parameters
    params = {
        "data": data_config,
        "cfg": "models/yolov5s.yaml",  # Use base model config
        "weights": "yolov5s-seg.pt",  # Pretrained segmentation weights
        "imgsz": 640,
        "batch_size": 8,
        "epochs": 50,
        "device": "cpu",  # Use GPU if available: 'cuda:0'
        "workers": 2,
        "project": "runs/train-seg",
        "name": "yolo_food_segmentation",
        "save_period": 10,
        "optimizer": "SGD",
        "lr0": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "cos_lr": True,
        "cache": False,
        "exist_ok": True,
        "verbose": True,
    }

    print("\n🔧 Training Parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")

    # Import YOLOv5 training module
    try:
        from segment.train import run

        print("\n🎯 Starting training...")
        print("📈 TensorBoard logs will be saved to: runs/train-seg/yolo_food_segmentation")

        # Start training
        run(**params)

        print("\n✅ Training completed!")
        print("📁 Results saved to: runs/train-seg/yolo_food_segmentation")
        print("🔍 Best weights: runs/train-seg/yolo_food_segmentation/weights/best.pt")
        print("📊 View results: tensorboard --logdir runs/train-seg/yolo_food_segmentation")

    except Exception as e:
        print(f"❌ Training error: {e}")
        print(f"💡 Try running with: python segment/train.py --data {data_config} --epochs 50 --batch-size 8")


def create_training_command():
    """Generate command line training command."""
    command = [
        "python",
        "segment/train.py",
        "--data",
        "yolo_dataset/data.yaml",
        "--cfg",
        "models/yolov5s.yaml",
        "--weights",
        "yolov5s-seg.pt",
        "--imgsz",
        "640",
        "--batch-size",
        "8",
        "--epochs",
        "50",
        "--device",
        "cpu",
        "--project",
        "runs/train-seg",
        "--name",
        "yolo_food_segmentation",
        "--save-period",
        "10",
        "--optimizer",
        "SGD",
        "--lr0",
        "0.01",
        "--momentum",
        "0.937",
        "--weight-decay",
        "0.0005",
        "--warmup-epochs",
        "3",
        "--cos-lr",
        "--cache",
        "--exist-ok",
    ]

    return " ".join(command)


if __name__ == "__main__":
    print("YOLOv5 Segmentation Training Script")
    print("=" * 40)

    # Check YOLO dataset
    if not os.path.exists("yolo_dataset/data.yaml"):
        print("❌ YOLO dataset not found!")
        print("💡 Run: python organize_yolo_dataset.py")
        sys.exit(1)

    print("✅ YOLO dataset found")

    # Show training command
    cmd = create_training_command()
    print("\n📋 Training Command:")
    print(f"   {cmd}")

    print("\n🚀 Choose training method:")
    print("   1. Run training function (recommended)")
    print("   2. Show command line only")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        train_yolo_segmentation()
    else:
        print("\n💻 Run this command:")
        print(f"   {cmd}")
        print("\n📊 View training progress:")
        print("   tensorboard --logdir runs/train-seg/yolo_food_segmentation")
