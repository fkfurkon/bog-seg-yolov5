#!/usr/bin/env python3
"""
Final Dataset Summary

This script provides a comprehensive summary of the merged dataset and training setup.
"""

import os
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

def analyze_annotations(labels_dir):
    """Analyze annotation files to get statistics"""
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        return {}
    
    class_counts = Counter()
    annotation_types = defaultdict(int)
    files_analyzed = 0
    
    for label_file in labels_path.glob("*.txt"):
        files_analyzed += 1
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    class_id = int(parts[0])
                    coords = parts[1:]
                    
                    class_counts[class_id] += 1
                    
                    # Determine annotation type
                    coord_count = len(coords)
                    if coord_count == 4:
                        annotation_types['bbox'] += 1
                    elif coord_count == 8:
                        annotation_types['simple_polygon'] += 1
                    elif coord_count > 8:
                        annotation_types['complex_polygon'] += 1
    
    return {
        'files_analyzed': files_analyzed,
        'class_counts': dict(class_counts),
        'annotation_types': dict(annotation_types),
        'total_annotations': sum(class_counts.values())
    }

def print_dataset_summary():
    """Print comprehensive dataset summary"""
    
    print("="*80)
    print("🎯 MERGED FOOD DATASET SUMMARY")
    print("="*80)
    
    # Check merged dataset
    merged_path = Path("merged_dataset_final")
    yolo_path = Path("yolo_dataset_final")
    
    if merged_path.exists():
        print(f"\n📁 Merged Dataset: {merged_path.absolute()}")
        
        # Read configuration
        data_yaml = merged_path / "data.yaml"
        if data_yaml.exists():
            with open(data_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"   Classes: {config.get('nc', 'Unknown')} classes")
            if 'names' in config:
                for idx, name in config['names'].items():
                    print(f"     {idx}: {name}")
    
    if yolo_path.exists():
        print(f"\n📁 YOLO Dataset: {yolo_path.absolute()}")
        
        # Read configuration
        data_yaml = yolo_path / "data.yaml"
        if data_yaml.exists():
            with open(data_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"   Dataset path: {config.get('path', 'Unknown')}")
            print(f"   Train path: {config.get('train', 'Unknown')}")
            print(f"   Val path: {config.get('val', 'Unknown')}")
            print(f"   Classes: {config.get('nc', 'Unknown')}")
            
            # Analyze train set
            train_labels = yolo_path / "labels/train"
            if train_labels.exists():
                print(f"\n📊 Training Set Analysis:")
                train_stats = analyze_annotations(train_labels)
                
                print(f"   Files: {train_stats['files_analyzed']}")
                print(f"   Total annotations: {train_stats['total_annotations']}")
                
                print(f"\n   🏷️  Class Distribution:")
                class_names = config.get('names', {})
                for class_id, count in sorted(train_stats['class_counts'].items()):
                    class_name = class_names.get(class_id, f"Class_{class_id}")
                    print(f"     {class_id}: {class_name} - {count} annotations")
                
                print(f"\n   📐 Annotation Types:")
                for ann_type, count in train_stats['annotation_types'].items():
                    print(f"     {ann_type}: {count}")
            
            # Analyze val set
            val_labels = yolo_path / "labels/val"
            if val_labels.exists():
                print(f"\n📊 Validation Set Analysis:")
                val_stats = analyze_annotations(val_labels)
                
                print(f"   Files: {val_stats['files_analyzed']}")
                print(f"   Total annotations: {val_stats['total_annotations']}")
                
                print(f"\n   🏷️  Class Distribution:")
                for class_id, count in sorted(val_stats['class_counts'].items()):
                    class_name = class_names.get(class_id, f"Class_{class_id}")
                    print(f"     {class_id}: {class_name} - {count} annotations")
    
    print(f"\n📋 TRAINING INSTRUCTIONS:")
    print(f"   1. Basic training:")
    print(f"      python train_final_merged_model.py")
    print(f"   2. Custom training:")
    print(f"      python train_final_merged_model.py --epochs 200 --batch-size 32")
    print(f"   3. Large model:")
    print(f"      python train_final_merged_model.py --model l --epochs 150")
    
    print(f"\n📝 FILES CREATED:")
    files_created = [
        "merge_bbox_seg_annotations.py",
        "organize_final_yolo_dataset.py", 
        "train_final_merged_model.py",
        "merged_dataset_final/",
        "yolo_dataset_final/"
    ]
    
    for file_name in files_created:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name}")
    
    print(f"\n🎯 DATASET READY FOR TRAINING!")
    print("="*80)

def main():
    print_dataset_summary()

if __name__ == "__main__":
    main()
