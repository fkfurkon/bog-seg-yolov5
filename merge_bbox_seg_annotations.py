#!/usr/bin/env python3
"""
Merge Bounding Box and Segmentation Annotations Script

This script merges two types of annotations:
1. Bounding box annotations (which appear to contain polygon data)
2. Segmentation annotations (which appear to contain bounding box data)

The script will:
- Read both annotation sets
- Merge them into a unified format
- Handle both bounding box and segmentation polygon formats
- Create a merged dataset with proper YOLO format annotations
"""

import os
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
import argparse

def read_annotation_file(file_path):
    """Read YOLO format annotation file and return list of annotations"""
    annotations = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    annotations.append(line)
    return annotations

def parse_annotation(annotation_line):
    """Parse a single annotation line"""
    parts = annotation_line.split()
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    return class_id, coords

def is_bbox_format(coords):
    """Check if coordinates are in bounding box format (4 values: x_center, y_center, width, height)"""
    return len(coords) == 4

def is_polygon_format(coords):
    """Check if coordinates are in polygon/segmentation format (multiple x,y pairs)"""
    return len(coords) > 4 and len(coords) % 2 == 0

def bbox_to_polygon(coords):
    """Convert bounding box to polygon format (4 corners)"""
    x_center, y_center, width, height = coords
    
    # Calculate corner coordinates
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center - height / 2
    x3 = x_center + width / 2
    y3 = y_center + height / 2
    x4 = x_center - width / 2
    y4 = y_center + height / 2
    
    # Return as polygon coordinates (x1,y1,x2,y2,x3,y3,x4,y4)
    return [x1, y1, x2, y2, x3, y3, x4, y4]

def polygon_to_bbox(coords):
    """Convert polygon coordinates to bounding box format"""
    x_coords = []
    y_coords = []
    
    # Extract x and y coordinates
    for i in range(0, len(coords), 2):
        x_coords.append(coords[i])
        y_coords.append(coords[i + 1])
    
    # Calculate bounding box
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Convert to YOLO format (center_x, center_y, width, height)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_center, y_center, width, height]

def merge_annotations(bbox_annotations, seg_annotations, output_format='both'):
    """
    Merge bounding box and segmentation annotations
    
    Args:
        bbox_annotations: List of bbox annotation lines
        seg_annotations: List of segmentation annotation lines
        output_format: 'bbox', 'polygon', or 'both'
    
    Returns:
        List of merged annotation lines
    """
    merged = []
    
    # Process bounding box annotations
    for ann in bbox_annotations:
        class_id, coords = parse_annotation(ann)
        
        if output_format == 'bbox':
            if is_polygon_format(coords):
                # Convert polygon to bbox
                bbox_coords = polygon_to_bbox(coords)
                merged.append(f"{class_id} {' '.join(map(str, bbox_coords))}")
            else:
                # Already bbox format
                merged.append(ann)
        elif output_format == 'polygon':
            if is_bbox_format(coords):
                # Convert bbox to polygon
                poly_coords = bbox_to_polygon(coords)
                merged.append(f"{class_id} {' '.join(map(str, poly_coords))}")
            else:
                # Already polygon format
                merged.append(ann)
        else:  # both
            merged.append(ann)
    
    # Process segmentation annotations
    for ann in seg_annotations:
        class_id, coords = parse_annotation(ann)
        
        if output_format == 'bbox':
            if is_polygon_format(coords):
                # Convert polygon to bbox
                bbox_coords = polygon_to_bbox(coords)
                merged.append(f"{class_id} {' '.join(map(str, bbox_coords))}")
            else:
                # Already bbox format
                merged.append(ann)
        elif output_format == 'polygon':
            if is_bbox_format(coords):
                # Convert bbox to polygon
                poly_coords = bbox_to_polygon(coords)
                merged.append(f"{class_id} {' '.join(map(str, poly_coords))}")
            else:
                # Already polygon format
                merged.append(ann)
        else:  # both
            merged.append(ann)
    
    return merged

def create_merged_dataset(bbox_root, seg_root, output_root, output_format='both'):
    """
    Create merged dataset from bounding box and segmentation annotations
    
    Args:
        bbox_root: Path to bounding box annotation directory
        seg_root: Path to segmentation annotation directory  
        output_root: Path to output merged dataset directory
        output_format: 'bbox', 'polygon', or 'both'
    """
    
    # Create output directories
    output_path = Path(output_root)
    output_path.mkdir(exist_ok=True)
    
    labels_dir = output_path / "labels" / "train"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Read data.yaml files
    bbox_yaml_path = Path(bbox_root) / "data.yaml"
    seg_yaml_path = Path(seg_root) / "data.yaml"
    
    # Use bbox data.yaml as base (they should be the same)
    if bbox_yaml_path.exists():
        with open(bbox_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
    elif seg_yaml_path.exists():
        with open(seg_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
    else:
        # Create default config
        data_config = {
            'names': {
                0: 'protein', 1: 'carbohydrate', 2: 'fruit', 3: 'dessert',
                4: 'flatware', 5: 'vegetable', 6: 'sauce', 7: 'soup', 8: 'snack'
            },
            'path': '.',
            'train': 'train.txt'
        }
    
    # Get all annotation files
    bbox_labels_dir = Path(bbox_root) / "labels" / "train"
    seg_labels_dir = Path(seg_root) / "labels" / "train"
    
    all_files = set()
    if bbox_labels_dir.exists():
        all_files.update([f.name for f in bbox_labels_dir.glob("*.txt")])
    if seg_labels_dir.exists():
        all_files.update([f.name for f in seg_labels_dir.glob("*.txt")])
    
    # Process each file
    merged_count = 0
    bbox_only_count = 0
    seg_only_count = 0
    
    for filename in sorted(all_files):
        bbox_file = bbox_labels_dir / filename
        seg_file = seg_labels_dir / filename
        
        # Read annotations from both sources
        bbox_annotations = read_annotation_file(bbox_file) if bbox_file.exists() else []
        seg_annotations = read_annotation_file(seg_file) if seg_file.exists() else []
        
        # Count statistics
        if bbox_annotations and seg_annotations:
            merged_count += 1
        elif bbox_annotations:
            bbox_only_count += 1
        elif seg_annotations:
            seg_only_count += 1
        
        # Merge annotations
        merged_annotations = merge_annotations(bbox_annotations, seg_annotations, output_format)
        
        # Write merged file
        if merged_annotations:
            output_file = labels_dir / filename
            with open(output_file, 'w') as f:
                for ann in merged_annotations:
                    f.write(ann + '\n')
    
    # Create train.txt file
    train_txt_path = output_path / "train.txt"
    with open(train_txt_path, 'w') as f:
        for filename in sorted(all_files):
            # Remove .txt extension and add image path
            image_name = filename.replace('.txt', '')
            f.write(f"../image/{image_name}.jpg\n")
    
    # Update data.yaml for merged dataset
    data_config['path'] = str(output_path)
    data_config['train'] = 'train.txt'
    
    output_yaml_path = output_path / "data.yaml"
    with open(output_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Merged dataset created at: {output_path}")
    print(f"Total files processed: {len(all_files)}")
    print(f"Files with both bbox and seg annotations: {merged_count}")
    print(f"Files with only bbox annotations: {bbox_only_count}")
    print(f"Files with only seg annotations: {seg_only_count}")
    print(f"Output format: {output_format}")

def main():
    parser = argparse.ArgumentParser(description='Merge bounding box and segmentation annotations')
    parser.add_argument('--bbox-root', 
                       default='dataset/annotation/boundingbox',
                       help='Path to bounding box annotation directory')
    parser.add_argument('--seg-root', 
                       default='dataset/annotation/segmentation',
                       help='Path to segmentation annotation directory')
    parser.add_argument('--output-root', 
                       default='merged_dataset',
                       help='Path to output merged dataset directory')
    parser.add_argument('--format', 
                       choices=['bbox', 'polygon', 'both'],
                       default='both',
                       help='Output annotation format')
    parser.add_argument('--preview', 
                       action='store_true',
                       help='Preview merge without creating output')
    
    args = parser.parse_args()
    
    if args.preview:
        print("Preview mode - analyzing annotations...")
        
        bbox_labels_dir = Path(args.bbox_root) / "labels" / "train"
        seg_labels_dir = Path(args.seg_root) / "labels" / "train"
        
        # Sample analysis
        sample_files = list(bbox_labels_dir.glob("*.txt"))[:3] if bbox_labels_dir.exists() else []
        
        for sample_file in sample_files:
            print(f"\nAnalyzing {sample_file.name}:")
            
            # Bbox annotations
            bbox_annotations = read_annotation_file(sample_file)
            print(f"  Bbox annotations: {len(bbox_annotations)}")
            for i, ann in enumerate(bbox_annotations[:2]):  # Show first 2
                class_id, coords = parse_annotation(ann)
                format_type = "bbox" if is_bbox_format(coords) else "polygon"
                print(f"    {i+1}. Class {class_id}: {format_type} ({len(coords)} coords)")
            
            # Seg annotations
            seg_file = seg_labels_dir / sample_file.name
            seg_annotations = read_annotation_file(seg_file) if seg_file.exists() else []
            print(f"  Seg annotations: {len(seg_annotations)}")
            for i, ann in enumerate(seg_annotations[:2]):  # Show first 2
                class_id, coords = parse_annotation(ann)
                format_type = "bbox" if is_bbox_format(coords) else "polygon"
                print(f"    {i+1}. Class {class_id}: {format_type} ({len(coords)} coords)")
        
    else:
        create_merged_dataset(args.bbox_root, args.seg_root, args.output_root, args.format)

if __name__ == "__main__":
    main()
