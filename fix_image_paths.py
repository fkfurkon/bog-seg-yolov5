#!/usr/bin/env python3
"""
Fix image paths in train.txt to match actual file extensions
"""

import os
from pathlib import Path

def fix_train_txt():
    """Fix the train.txt file to use correct image extensions"""
    
    # Paths
    dataset_root = Path("/home/korn/study/y4-1/fern/lab/Lab5-Train-Box-Seg-v4/yolov5")
    merged_dataset = dataset_root / "merged_dataset"
    image_dir = dataset_root / "dataset" / "image"
    train_txt = merged_dataset / "train.txt"
    
    # Get all image files with their actual extensions
    image_files = {}
    for img_file in image_dir.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Get the base name without extension
            base_name = img_file.stem
            image_files[base_name] = img_file.name
    
    print(f"Found {len(image_files)} image files")
    
    # Read current train.txt
    with open(train_txt, 'r') as f:
        lines = f.readlines()
    
    # Fix the paths
    new_lines = []
    fixed_count = 0
    
    for line in lines:
        line = line.strip()
        if line:
            # Extract the base name from the current path
            # Current format: ../image/1.jpg
            base_name = Path(line).stem
            
            if base_name in image_files:
                # Use the actual image path relative to the dataset
                actual_filename = image_files[base_name]
                new_path = f"../dataset/image/{actual_filename}"
                new_lines.append(new_path + '\n')
                fixed_count += 1
            else:
                print(f"Warning: No image found for {base_name}")
                new_lines.append(line + '\n')
    
    # Write the fixed train.txt
    with open(train_txt, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Fixed {fixed_count} image paths in train.txt")
    print(f"Updated {train_txt}")

if __name__ == "__main__":
    fix_train_txt()
