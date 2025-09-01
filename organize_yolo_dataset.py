#!/usr/bin/env python
"""
YOLO Dataset Format Organizer
This script organizes the dataset into proper YOLO training format
"""

import os
import shutil
from pathlib import Path
import yaml

def create_yolo_format_dataset():
    """Create YOLO format dataset structure"""
    
    print("🔄 Organizing dataset into YOLO format...")
    
    # Create YOLO format directory structure
    yolo_dir = Path('yolo_dataset')
    yolo_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (yolo_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (yolo_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (yolo_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (yolo_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Source directories
    source_images = Path('dataset/image')
    source_labels = Path('dataset/annotation/boundingbox/labels/train')
    
    # Get all images
    image_files = list(source_images.glob('*.jpg')) + list(source_images.glob('*.png'))
    print(f"📁 Found {len(image_files)} images")
    
    # Get all label files
    label_files = list(source_labels.glob('*.txt'))
    print(f"🏷️  Found {len(label_files)} label files")
    
    # Split data (80% train, 20% val)
    train_split = 0.8
    train_count = int(len(image_files) * train_split)
    
    train_images = image_files[:train_count]
    val_images = image_files[train_count:]
    
    print(f"📊 Train: {len(train_images)} images, Val: {len(val_images)} images")
    
    # Copy train images and labels
    train_list = []
    for img_path in train_images:
        # Copy image
        img_name = img_path.name
        img_stem = img_path.stem
        dest_img = yolo_dir / 'images' / 'train' / img_name
        shutil.copy2(img_path, dest_img)
        train_list.append(f"./images/train/{img_name}")
        
        # Copy corresponding label if exists
        label_path = source_labels / f"{img_stem}.txt"
        if label_path.exists():
            dest_label = yolo_dir / 'labels' / 'train' / f"{img_stem}.txt"
            shutil.copy2(label_path, dest_label)
    
    # Copy val images and labels
    val_list = []
    for img_path in val_images:
        # Copy image
        img_name = img_path.name
        img_stem = img_path.stem
        dest_img = yolo_dir / 'images' / 'val' / img_name
        shutil.copy2(img_path, dest_img)
        val_list.append(f"./images/val/{img_name}")
        
        # Copy corresponding label if exists
        label_path = source_labels / f"{img_stem}.txt"
        if label_path.exists():
            dest_label = yolo_dir / 'labels' / 'val' / f"{img_stem}.txt"
            shutil.copy2(label_path, dest_label)
    
    # Create train.txt and val.txt files
    with open(yolo_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(train_list))
    
    with open(yolo_dir / 'val.txt', 'w') as f:
        f.write('\n'.join(val_list))
    
    # Create YOLO dataset configuration
    dataset_config = {
        'path': str(yolo_dir.absolute()),
        'train': 'train.txt',
        'val': 'val.txt',
        'nc': 9,  # number of classes
        'names': {
            0: 'protein',
            1: 'carbohydrate', 
            2: 'fruit',
            3: 'dessert',
            4: 'flatware',
            5: 'vegetable',
            6: 'sauce',
            7: 'soup',
            8: 'snack'
        }
    }
    
    # Save YOLO dataset config
    with open(yolo_dir / 'data.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ YOLO dataset created at: {yolo_dir.absolute()}")
    print(f"📁 Structure:")
    print(f"   ├── images/")
    print(f"   │   ├── train/ ({len(train_images)} images)")
    print(f"   │   └── val/ ({len(val_images)} images)")
    print(f"   ├── labels/")
    print(f"   │   ├── train/ (annotation files)")
    print(f"   │   └── val/ (annotation files)")
    print(f"   ├── data.yaml (dataset configuration)")
    print(f"   ├── train.txt (training file list)")
    print(f"   └── val.txt (validation file list)")
    
    return yolo_dir

if __name__ == '__main__':
    yolo_dir = create_yolo_format_dataset()
    print(f"\n🚀 Ready for YOLO training with: {yolo_dir}/data.yaml")
