#!/usr/bin/env python3
"""
Organize Final YOLO Dataset

This script organizes the merged dataset into a proper YOLO training format with:
- Train/validation splits
- Proper directory structure
- Updated data.yaml configuration
- Image-label pairing verification
"""

import os
import shutil
import yaml
import random
from pathlib import Path
import argparse

def create_yolo_structure(output_root):
    """Create proper YOLO directory structure"""
    output_path = Path(output_root)
    
    # Create main directories
    dirs_to_create = [
        "images/train",
        "images/val", 
        "labels/train",
        "labels/val"
    ]
    
    for dir_name in dirs_to_create:
        (output_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    return output_path

def get_image_files(image_dir):
    """Get all image files from directory"""
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_files = []
    if image_dir.exists():
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)

def split_dataset(image_files, val_ratio=0.2, seed=42):
    """Split dataset into train and validation sets"""
    random.seed(seed)
    
    # Shuffle the files
    shuffled_files = image_files.copy()
    random.shuffle(shuffled_files)
    
    # Calculate split point
    val_count = int(len(shuffled_files) * val_ratio)
    
    val_files = shuffled_files[:val_count]
    train_files = shuffled_files[val_count:]
    
    return train_files, val_files

def copy_files_with_labels(file_list, source_image_dir, source_label_dir, 
                          dest_image_dir, dest_label_dir, split_name):
    """Copy image and corresponding label files"""
    copied_count = 0
    missing_labels = []
    
    for image_file in file_list:
        # Copy image
        dest_image_path = Path(dest_image_dir) / image_file.name
        shutil.copy2(image_file, dest_image_path)
        
        # Find corresponding label file
        label_name = image_file.stem + '.txt'
        source_label_path = Path(source_label_dir) / label_name
        
        if source_label_path.exists():
            dest_label_path = Path(dest_label_dir) / label_name
            shutil.copy2(source_label_path, dest_label_path)
            copied_count += 1
        else:
            missing_labels.append(image_file.name)
    
    print(f"{split_name} set: {len(file_list)} images, {copied_count} labels")
    if missing_labels:
        print(f"  Warning: {len(missing_labels)} images without labels")
        if len(missing_labels) <= 5:
            print(f"  Missing labels for: {missing_labels}")
    
    return copied_count, missing_labels

def create_dataset_yaml(output_root, class_names, train_count, val_count):
    """Create data.yaml file for YOLO training"""
    
    data_config = {
        'path': str(Path(output_root).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = Path(output_root) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nDataset configuration saved to: {yaml_path}")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {list(class_names.values())}")

def verify_dataset(output_root):
    """Verify the organized dataset"""
    output_path = Path(output_root)
    
    # Count files
    train_images = len(list((output_path / "images/train").glob("*")))
    train_labels = len(list((output_path / "labels/train").glob("*.txt")))
    val_images = len(list((output_path / "images/val").glob("*")))
    val_labels = len(list((output_path / "labels/val").glob("*.txt")))
    
    print(f"\nDataset Verification:")
    print(f"Train: {train_images} images, {train_labels} labels")
    print(f"Val: {val_images} images, {val_labels} labels")
    
    # Check for mismatches
    train_mismatch = train_images - train_labels
    val_mismatch = val_images - val_labels
    
    if train_mismatch != 0:
        print(f"⚠️  Train set mismatch: {train_mismatch} images without labels")
    if val_mismatch != 0:
        print(f"⚠️  Val set mismatch: {val_mismatch} images without labels")
    
    if train_mismatch == 0 and val_mismatch == 0:
        print("✅ All images have corresponding labels")
    
    return train_images, val_images

def organize_yolo_dataset(merged_dataset_path, image_dir, output_root, val_ratio=0.2):
    """Main function to organize the YOLO dataset"""
    
    print(f"Organizing YOLO dataset...")
    print(f"Merged annotations: {merged_dataset_path}")
    print(f"Images directory: {image_dir}")
    print(f"Output directory: {output_root}")
    print(f"Validation ratio: {val_ratio}")
    
    # Create output structure
    output_path = create_yolo_structure(output_root)
    
    # Read class names from merged dataset
    merged_yaml = Path(merged_dataset_path) / 'data.yaml'
    if merged_yaml.exists():
        with open(merged_yaml, 'r') as f:
            merged_config = yaml.safe_load(f)
            class_names = merged_config.get('names', {})
    else:
        # Default classes
        class_names = {
            0: 'protein', 1: 'carbohydrate', 2: 'fruit', 3: 'dessert',
            4: 'flatware', 5: 'vegetable', 6: 'sauce', 7: 'soup', 8: 'snack'
        }
    
    # Get all image files
    image_files = get_image_files(image_dir)
    print(f"\nFound {len(image_files)} image files")
    
    if not image_files:
        print("❌ No image files found!")
        return
    
    # Split dataset
    train_files, val_files = split_dataset(image_files, val_ratio)
    print(f"Split: {len(train_files)} train, {len(val_files)} val")
    
    # Source label directory
    source_label_dir = Path(merged_dataset_path) / "labels/train"
    
    # Copy train files
    train_count, train_missing = copy_files_with_labels(
        train_files, image_dir, source_label_dir,
        output_path / "images/train", output_path / "labels/train", "Train"
    )
    
    # Copy val files
    val_count, val_missing = copy_files_with_labels(
        val_files, image_dir, source_label_dir,
        output_path / "images/val", output_path / "labels/val", "Val"
    )
    
    # Create data.yaml
    create_dataset_yaml(output_root, class_names, train_count, val_count)
    
    # Verify dataset
    verify_dataset(output_root)
    
    print(f"\n✅ YOLO dataset organized successfully!")
    print(f"📁 Dataset location: {output_path.absolute()}")
    print(f"📄 Start training with: python train.py --data {output_path.absolute()}/data.yaml")

def main():
    parser = argparse.ArgumentParser(description='Organize merged dataset into YOLO format')
    parser.add_argument('--merged-dataset', 
                       default='merged_dataset_final',
                       help='Path to merged dataset directory')
    parser.add_argument('--image-dir', 
                       default='dataset/image',
                       help='Path to images directory')
    parser.add_argument('--output', 
                       default='yolo_dataset_final',
                       help='Output directory for organized YOLO dataset')
    parser.add_argument('--val-ratio', 
                       type=float, 
                       default=0.2,
                       help='Validation set ratio (default: 0.2)')
    parser.add_argument('--seed', 
                       type=int, 
                       default=42,
                       help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    organize_yolo_dataset(
        args.merged_dataset,
        args.image_dir, 
        args.output,
        args.val_ratio
    )

if __name__ == "__main__":
    main()
